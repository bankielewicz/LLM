from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

from llm_foundations_companion.database import Database
from llm_foundations_companion.errors import ApiError
from llm_foundations_companion.registry import ROOT_QUOTA_BYTES, Registry
from llm_foundations_companion.schema import canonical_json
from llm_foundations_companion.training_store import (
    AdmissionPlan,
    TrainingStore,
    _checked_u64,
)


def _runtime(
    tmp_path: Path,
    *,
    revision: str | None = "2" * 40,
    uuid_factory: Any = None,
) -> tuple[Database, Registry, TrainingStore]:
    root = tmp_path / str(uuid.uuid4())
    database = Database(root)
    database.initialize()
    registry = Registry(database, root, str(uuid.uuid4()), os.urandom(32))
    store = TrainingStore(
        database,
        registry,
        runtime_profile="wsl-cpu",
        device="cpu",
        dependency_lock_sha256="1" * 64,
        companion_source_revision=revision,
        uuid_factory=uuid_factory,
    )
    return database, registry, store


def _plan(resolved: dict[str, Any], *, operation: str = "context_preview") -> AdmissionPlan:
    return AdmissionPlan(
        operation=operation,
        reservation={
            "byte_count": 1024,
            "artifact_rows": 1,
            "dataset_rows": 0,
            "run_rows": 0,
            "model_rows": 0,
            "checkpoint_rows": 0,
        },
        resolved=resolved,
        run_id=None,
        model_id=None,
        tokenizer_id=None,
        checkpoint_ids=(),
    )


def _assert_storage_corrupt(exc: ApiError) -> None:
    assert exc.code == "STORAGE_UNAVAILABLE"
    assert exc.reason_code == "STORAGE_CORRUPT"


def test_stored_malformed_document_text_is_storage_corruption() -> None:
    with pytest.raises(ApiError) as caught:
        TrainingStore._document_text_bytes(b'{')

    _assert_storage_corrupt(caught.value)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"runtime_profile": "other"}, "runtime_profile"),
        ({"device": "cuda"}, "device"),
        ({"dependency_lock_sha256": "A" * 64}, "dependency_lock_sha256"),
        ({"companion_source_revision": "2" * 39}, "companion_source_revision"),
    ],
)
def test_training_store_rejects_invalid_runtime_identity(
    tmp_path: Path, overrides: dict[str, str], message: str
) -> None:
    database, registry, _ = _runtime(tmp_path)
    values = {
        "runtime_profile": "wsl-cpu",
        "device": "cpu",
        "dependency_lock_sha256": "1" * 64,
        "companion_source_revision": "2" * 40,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        TrainingStore(database, registry, **values)


def test_identifier_and_runtime_provenance_guards(tmp_path: Path) -> None:
    noncanonical = str(uuid.uuid4()).upper()
    _, _, store = _runtime(tmp_path, uuid_factory=lambda: noncanonical)
    with pytest.raises(ValueError, match="noncanonical"):
        store._new_id()

    _, _, unbound = _runtime(tmp_path, revision=None)
    with pytest.raises(ApiError) as caught:
        unbound._require_runtime_identity()
    assert caught.value.code == "CAPABILITY_UNAVAILABLE"


def test_checked_arithmetic_and_tiny_reservation_bounds(tmp_path: Path) -> None:
    _, _, store = _runtime(tmp_path)
    assert _checked_u64(0, "/steps") == 0
    assert _checked_u64((1 << 64) - 1, "/steps") == (1 << 64) - 1
    for value in (True, -1, 1 << 64):
        with pytest.raises(ApiError) as caught:
            _checked_u64(value, "/steps")
        assert caught.value.reason_code == "ESTIMATE_EXCEEDS_ROOT_QUOTA"

    reservation, boundaries = store._tiny_reservation(
        parameter_count=257, steps=1, eval_every=1, field_path="/eval_every"
    )
    assert boundaries == 3
    assert reservation == {
        "byte_count": 512 * 1024 * 1024,
        "artifact_rows": 29,
        "dataset_rows": 0,
        "run_rows": 1,
        "model_rows": 1,
        "checkpoint_rows": 3,
    }
    with pytest.raises(ApiError) as caught:
        store._tiny_reservation(
            parameter_count=10_000_000,
            steps=2_147_483_647,
            eval_every=1,
            field_path="/additional_steps",
        )
    assert caught.value.reason_code == "ESTIMATE_EXCEEDS_ROOT_QUOTA"
    assert ROOT_QUOTA_BYTES < (2 + 2_147_483_647) * (120_000_000 + 8 * 1024 * 1024)


def test_document_text_and_dataset_policy_validation() -> None:
    assert TrainingStore._document_text_bytes(
        b'{"text":"caf\xc3\xa9"}\n{"text":""}\n'
    ) == 5
    for raw in (b"", b'{"record_id":"x"}\n', b"[]\n"):
        with pytest.raises(ApiError) as caught:
            TrainingStore._document_text_bytes(raw)
        _assert_storage_corrupt(caught.value)

    with pytest.raises(ApiError) as caught:
        TrainingStore._require_tiny_dataset(
            {"format": "llm-foundations-dataset-v1", "record_format": "retrieval_v1"},
            "/dataset_id",
        )
    assert caught.value.reason_code == "SUBJECT_INCOMPATIBLE"
    with pytest.raises(ApiError) as caught:
        TrainingStore._require_eligible_dataset(
            {"eligibility": "audit_only"}, "/dataset_id"
        )
    assert caught.value.reason_code == "DATASET_INELIGIBLE"
    with pytest.raises(ApiError) as caught:
        TrainingStore._enforce_text_cap(
            [{"split": "train", "text_bytes": 200_001}], {"train"}, "/dataset_id"
        )
    assert caught.value.reason_code == "PAYLOAD_TOO_LARGE"


def test_tiny_tokenizer_parser_binds_exact_serialization() -> None:
    value = {"format": "teaching-byte-bpe-v1", "merges": [[0, 1], [257, 2]]}
    raw = json.dumps(value, indent=2).encode("utf-8")
    normalized, fingerprint, vocab_size = TrainingStore._tiny_tokenizer_bytes(raw)
    assert normalized == value
    assert vocab_size == 259
    assert fingerprint == hashlib.sha256(
        json.dumps(value, sort_keys=True).encode("utf-8")
    ).hexdigest()

    invalid = [
        json.dumps(value, separators=(",", ":")).encode("utf-8"),
        json.dumps({"format": "teaching-byte-bpe-v1", "merges": [[True, 1]]}, indent=2).encode(),
        json.dumps({"format": "teaching-byte-bpe-v1", "merges": [[256, 1]]}, indent=2).encode(),
        json.dumps({"format": "teaching-byte-bpe-v1", "merges": [[257, 1]]}, indent=2).encode(),
    ]
    for payload in invalid:
        with pytest.raises(ApiError) as caught:
            TrainingStore._tiny_tokenizer_bytes(payload)
        _assert_storage_corrupt(caught.value)


def test_tiny_config_parser_enforces_shape_and_parameter_cap() -> None:
    base = {
        "format": "tiny-v2-config-v1",
        "architecture_profile_id": "tiny-v2-standard-v1",
        "vocab_size": 257,
        "context": 8,
        "width": 16,
        "heads": 1,
        "layers": 1,
    }
    assert TrainingStore._tiny_config_bytes(json.dumps(base).encode()) == base
    invalid = []
    for changes in (
        {"context": True},
        {"width": 17, "heads": 2},
        {"width": 512, "heads": 16, "layers": 12, "vocab_size": 1024, "context": 512},
        {"architecture_profile_id": "unknown"},
    ):
        value = dict(base)
        value.update(changes)
        invalid.append(json.dumps(value).encode())
    for payload in invalid:
        with pytest.raises(ApiError) as caught:
            TrainingStore._tiny_config_bytes(payload)
        _assert_storage_corrupt(caught.value)


@pytest.mark.parametrize(
    ("job_request", "paths"),
    [
        (
            {
                "operation": "generate",
                "checkpoint_id": "checkpoint",
                "preview_artifact_id": "preview",
            },
            ["/checkpoint_id", "/preview_artifact_id"],
        ),
        (
            {
                "operation": "context_preview",
                "backend": "applied",
                "subject": {"model_id": "model", "adapter_checkpoint_id": "adapter"},
            },
            ["/subject/adapter_checkpoint_id", "/subject/model_id"],
        ),
        (
            {
                "operation": "evaluate",
                "dataset_id": "dataset",
                "split": "test",
                "subjects": [
                    {"kind": "base_model", "model_id": "model"},
                    {"kind": "tiny_checkpoint", "checkpoint_id": "checkpoint"},
                ],
            },
            ["/dataset_id", "/subjects/0/model_id", "/subjects/1/checkpoint_id"],
        ),
    ],
)
def test_reference_collection_reports_all_missing_resources(
    tmp_path: Path, job_request: dict[str, Any], paths: list[str]
) -> None:
    _, _, store = _runtime(tmp_path)
    with pytest.raises(ApiError) as caught:
        store._assert_request_references(job_request)
    assert caught.value.reason_code == "REFERENCE_MISSING"
    assert [item["field_path"] for item in caught.value.field_errors] == paths


def test_reference_collection_reports_existing_dataset_missing_required_split(
    tmp_path: Path,
) -> None:
    database, registry, store = _runtime(tmp_path)
    artifact = registry.register_stream(
        io.BytesIO(b"audit"),
        str(uuid.uuid4()),
        artifact_type="dataset_audit",
        display_name="audit.json",
        media_type="application/json",
        preview_policy="metadata_only",
        origin="locally_created",
    )
    dataset_id = str(uuid.uuid4())
    now = "2026-10-01T00:00:00.000Z"
    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO datasets(
                dataset_id, format, name, record_format, origin, audit_artifact_id,
                created_at, provenance_note, manifest_sha256, eligibility,
                manifest_json, source_identity_json, updated_at
            ) VALUES (?, 'llm-foundations-dataset-v1', 'missing-split',
                      'document_text_v1', 'locally_created', ?, ?, NULL, ?,
                      'eligible', '{}', NULL, ?)
            """,
            (dataset_id, artifact["artifact_id"], now, "3" * 64, now),
        )
    with pytest.raises(ApiError) as caught:
        store._assert_request_references(
            {"operation": "tokenizer_train", "dataset_id": dataset_id}
        )
    assert [item["field_path"] for item in caught.value.field_errors] == ["/dataset_id"]


def test_context_read_rejects_missing_and_structurally_corrupt_rows(tmp_path: Path) -> None:
    database, _, store = _runtime(tmp_path)
    with pytest.raises(ApiError) as missing:
        store.get_job_context(str(uuid.uuid4()))
    assert missing.value.code == "NOT_FOUND"

    job_id = str(uuid.uuid4())
    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO job_operation_contexts(
                job_id, operation, request_sha256, runtime_profile, device,
                dependency_lock_sha256, companion_source_revision, run_id,
                model_id, tokenizer_id, checkpoint_ids_json, reservation_json,
                resolved_json, created_at
            ) VALUES (?, 'context_preview', ?, 'wsl-cpu', 'cpu', ?, ?, NULL,
                      NULL, NULL, '[]', '{}', '[]', ?)
            """,
            (job_id, "3" * 64, "1" * 64, "2" * 40, "2026-10-01T00:00:00.000Z"),
        )
    with pytest.raises(ApiError) as corrupt:
        store.get_job_context(job_id)
    _assert_storage_corrupt(corrupt.value)


def test_snapshot_rejects_corrupt_input_descriptors(tmp_path: Path) -> None:
    database, _, store = _runtime(tmp_path)
    cases: list[Any] = [
        {},
        [1],
        [{"role": "bad/role"}],
        [
            {
                "role": "dataset.train",
                "artifact_id": str(uuid.uuid4()),
                "sha256": "bad",
                "size_bytes": 1,
                "sealed": False,
            }
        ],
    ]
    for raw_inputs in cases:
        job_id = str(uuid.uuid4())
        with database.transaction() as connection:
            store.create_job_context_in(
                connection,
                job_id=job_id,
                request_sha256="3" * 64,
                plan=_plan({"request": {}, "_inputs": raw_inputs}),
            )
        with pytest.raises(ApiError) as caught:
            store.materialize_worker_snapshot(job_id)
        _assert_storage_corrupt(caught.value)


def test_verified_copy_is_idempotent_and_cleans_failed_partial(tmp_path: Path) -> None:
    raw = b"verified"
    digest = hashlib.sha256(raw).hexdigest()
    destination = tmp_path / "copy.bin"
    TrainingStore._write_verified_copy(
        destination, io.BytesIO(raw), expected_size=len(raw), expected_sha256=digest
    )
    TrainingStore._write_verified_copy(
        destination, io.BytesIO(b"ignored"), expected_size=len(raw), expected_sha256=digest
    )
    assert destination.read_bytes() == raw

    destination.write_bytes(b"tampered")
    with pytest.raises(ApiError) as caught:
        TrainingStore._write_verified_copy(
            destination, io.BytesIO(raw), expected_size=len(raw), expected_sha256=digest
        )
    _assert_storage_corrupt(caught.value)

    failed = tmp_path / "failed.bin"
    with pytest.raises(OSError, match="digest mismatch"):
        TrainingStore._write_verified_copy(
            failed, io.BytesIO(raw), expected_size=len(raw), expected_sha256="f" * 64
        )
    assert not failed.exists()
    assert not list(tmp_path.glob(".failed.bin.*.partial"))


def _resume_admission_case() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    dataset_id = str(uuid.uuid4())
    checkpoint_id = str(uuid.uuid4())
    tokenizer_sha256 = "4" * 64
    splits = [
        {
            "split": name,
            "dataset_id": dataset_id,
            "artifact_id": str(uuid.uuid4()),
            "sha256": digest,
            "sealed": False,
            "text_bytes": 1,
        }
        for name, digest in (("train", "5" * 64), ("validation", "6" * 64))
    ]
    checkpoint = {
        "checkpoint_id": checkpoint_id,
        "step": 1,
        "seed": 17,
        "model_identity": {
            "architecture_profile_id": "tiny-v2-standard-v1",
            "tokenizer_sha256": tokenizer_sha256,
        },
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
        "dataset_bindings": [
            {
                "split": item["split"],
                "dataset_id": dataset_id,
                "artifact_id": item["artifact_id"],
                "sha256": item["sha256"],
            }
            for item in splits
        ],
        "_files": {"trainer_state.json": {"artifact_id": str(uuid.uuid4())}},
    }
    trainer = {
        "format": "tiny-v2-trainer-state-v1",
        "completed_global_step": 1,
        "requested_final_step": 1,
        "eval_every": 1,
        "batch_size": 1,
        "seed": 17,
        "learning_rate": 0.001,
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.999],
            "eps": 1e-08,
            "weight_decay": 0.01,
        },
        "parameter_steps": {"model.weight": 1},
        "incumbent": {
            "checkpoint_id": None,
            "validation_nll_token": None,
            "step": None,
        },
        "dataset_bindings": [
            {
                "split": item["split"],
                "dataset_id": dataset_id,
                "sha256": item["sha256"],
            }
            for item in splits
        ],
        "tokenizer_sha256": tokenizer_sha256,
        "architecture_profile_id": "tiny-v2-standard-v1",
        "runtime_profile": "wsl-cpu",
        "device": "cpu",
        "dependency_lock_sha256": "1" * 64,
    }
    config = {
        "format": "tiny-v2-config-v1",
        "architecture_profile_id": "tiny-v2-standard-v1",
        "vocab_size": 257,
        "context": 8,
        "width": 16,
        "heads": 1,
        "layers": 1,
    }
    return checkpoint, trainer, config, splits


@pytest.mark.parametrize(
    ("case", "reason_code"),
    [
        ("malformed_trainer", "CHECKPOINT_INCOMPATIBLE"),
        ("runtime_identity", "CHECKPOINT_INCOMPATIBLE"),
        ("step_overflow", "TINY_STEP_OVERFLOW"),
        ("config_identity", "CHECKPOINT_INCOMPATIBLE"),
        ("dataset_shape", "CHECKPOINT_INCOMPATIBLE"),
        ("dataset_drift", "CHECKPOINT_INCOMPATIBLE"),
    ],
)
def test_resume_admission_rejects_incompatible_checkpoint_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, reason_code: str
) -> None:
    _, _, store = _runtime(tmp_path)
    checkpoint, trainer, config, splits = _resume_admission_case()
    raw_trainer = canonical_json(trainer)
    if case == "malformed_trainer":
        raw_trainer = b"{"
    elif case == "runtime_identity":
        trainer["dependency_lock_sha256"] = "7" * 64
        raw_trainer = canonical_json(trainer)
    elif case == "step_overflow":
        checkpoint["step"] = 2_147_483_647
        trainer["completed_global_step"] = 2_147_483_647
        trainer["requested_final_step"] = 2_147_483_647
        raw_trainer = canonical_json(trainer)
    elif case == "config_identity":
        config["width"] = 32
    elif case == "dataset_shape":
        checkpoint["dataset_bindings"] = checkpoint["dataset_bindings"][:1]
    elif case == "dataset_drift":
        splits[0]["sha256"] = "8" * 64

    monkeypatch.setattr(store, "_assert_request_references", lambda _value: None)
    monkeypatch.setattr(
        store, "_checkpoint", lambda _checkpoint_id, require_resume: (checkpoint, [])
    )
    monkeypatch.setattr(store.registry, "read_artifact_bytes", lambda _artifact_id: raw_trainer)
    monkeypatch.setattr(
        store,
        "_checkpoint_tiny_identity",
        lambda _checkpoint: (config, "4" * 64, 257),
    )
    monkeypatch.setattr(
        store,
        "_dataset",
        lambda _dataset_id, _splits, *, field_path: (
            {"manifest_sha256": "9" * 64},
            splits,
        ),
    )
    monkeypatch.setattr(store, "_require_tiny_dataset", lambda *_args: None)
    monkeypatch.setattr(store, "_enforce_text_cap", lambda *_args: None)
    monkeypatch.setattr(store, "_require_eligible_dataset", lambda *_args: None)
    monkeypatch.setattr(
        store,
        "_input",
        lambda role, artifact_id, sealed=False: {
            "role": role,
            "artifact_id": artifact_id,
            "sealed": sealed,
        },
    )

    with pytest.raises(ApiError) as caught:
        store.resolve(
            {
                "operation": "tiny_resume",
                "checkpoint_id": checkpoint["checkpoint_id"],
                "additional_steps": 1,
            }
        )
    assert caught.value.reason_code == reason_code


@pytest.mark.parametrize(
    ("profile", "subjects", "sealed", "reason_code"),
    [
        (
            "applied-intents-greedy-v1",
            [{"kind": "base_model", "model_id": "MODEL_ID"}],
            False,
            "SUBJECT_INCOMPATIBLE",
        ),
        (
            "tiny-nll-per-byte-v1",
            [{"kind": "tiny_checkpoint", "checkpoint_id": "CHECKPOINT_ID"}],
            True,
            "SEALED_SPLIT_REQUIRES_TOKEN",
        ),
    ],
)
def test_evaluate_admission_enforces_slice_and_sealed_split_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    subjects: list[dict[str, str]],
    sealed: bool,
    reason_code: str,
) -> None:
    _, _, store = _runtime(tmp_path)
    dataset_id = str(uuid.uuid4())
    for subject in subjects:
        for key, value in tuple(subject.items()):
            if value in {"MODEL_ID", "CHECKPOINT_ID"}:
                subject[key] = str(uuid.uuid4())
    monkeypatch.setattr(store, "_assert_request_references", lambda _value: None)
    monkeypatch.setattr(
        store,
        "_dataset",
        lambda *_args, **_kwargs: (
            {"manifest_sha256": "a" * 64},
            [
                {
                    "split": "validation",
                    "dataset_id": dataset_id,
                    "artifact_id": str(uuid.uuid4()),
                    "sha256": "b" * 64,
                    "sealed": sealed,
                    "text_bytes": 1,
                }
            ],
        ),
    )
    monkeypatch.setattr(
        store, "_checkpoint_resolution", lambda _checkpoint_id: ({}, [])
    )
    monkeypatch.setattr(store, "_require_tiny_dataset", lambda *_args: None)
    monkeypatch.setattr(store, "_enforce_text_cap", lambda *_args: None)
    monkeypatch.setattr(store, "_require_eligible_dataset", lambda *_args: None)

    with pytest.raises(ApiError) as caught:
        store.resolve(
            {
                "operation": "evaluate",
                "subjects": subjects,
                "dataset_id": dataset_id,
                "split": "validation",
                "evaluation_profile_id": profile,
            }
        )
    assert caught.value.reason_code == reason_code


def _tiny_preview_request(checkpoint_id: str) -> dict[str, Any]:
    return {
        "operation": "context_preview",
        "backend": "tiny",
        "checkpoint_id": checkpoint_id,
        "prompt": "hello",
        "max_new_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 17,
    }


def test_context_preview_admission_requires_exact_checkpoint_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, store = _runtime(tmp_path)
    checkpoint_id = str(uuid.uuid4())
    monkeypatch.setattr(store, "_assert_request_references", lambda _value: None)
    inputs = [
        {"role": "checkpoint.tokenizer.json", "artifact_id": str(uuid.uuid4())},
        {"role": "checkpoint.config.json", "artifact_id": str(uuid.uuid4())},
        {"role": "checkpoint.model.safetensors", "artifact_id": str(uuid.uuid4())},
    ]
    monkeypatch.setattr(
        store,
        "_checkpoint_resolution",
        lambda _checkpoint_id: ({"checkpoint_id": checkpoint_id}, inputs),
    )
    plan = store.resolve(_tiny_preview_request(checkpoint_id))
    assert [item["role"] for item in plan.resolved["_inputs"]] == [
        "checkpoint.tokenizer.json",
        "checkpoint.config.json",
    ]

    inputs.pop(1)
    with pytest.raises(ApiError) as caught:
        store.resolve(_tiny_preview_request(checkpoint_id))
    _assert_storage_corrupt(caught.value)


def test_generate_admission_binds_canonical_preview_to_exact_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, registry, store = _runtime(tmp_path)
    checkpoint_id = str(uuid.uuid4())
    checkpoint = {
        "checkpoint_id": checkpoint_id,
        "model_id": str(uuid.uuid4()),
        "checkpoint_sha256": "c" * 64,
        "tokenizer_sha256": "d" * 64,
        "context": 8,
    }
    generation_request = {
        "operation": "generate",
        "checkpoint_id": checkpoint_id,
        "prompt": "hello",
        "max_new_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 17,
    }
    subject_identity = {
        "backend": "tiny",
        "model_id": checkpoint["model_id"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "runtime_profile": "wsl-cpu",
        "device": "cpu",
        "context_limit": 8,
    }
    preview = {
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
        "effective_context_budget": 8,
        "canonical_generation_request_sha256": hashlib.sha256(
            canonical_json(generation_request)
        ).hexdigest(),
    }
    preview_bytes = canonical_json(preview)
    preview_artifact = registry.register_stream(
        io.BytesIO(preview_bytes),
        str(uuid.uuid4()),
        artifact_type="context_preview",
        display_name="preview.json",
        media_type="application/json",
        preview_policy="metadata_only",
        origin="locally_created",
    )
    digest = hashlib.sha256(preview_bytes).hexdigest()
    request = {
        **generation_request,
        "preview_artifact_id": preview_artifact["artifact_id"],
        "context_preview_digest": digest,
    }
    monkeypatch.setattr(store, "_assert_request_references", lambda _value: None)
    monkeypatch.setattr(
        store,
        "_checkpoint_resolution",
        lambda _checkpoint_id: (checkpoint, []),
    )

    plan = store.resolve(request)
    assert plan.resolved["preview"] == preview
    assert plan.resolved["subject_identity"] == subject_identity

    stale_request = copy.deepcopy(request)
    stale_request["context_preview_digest"] = "e" * 64
    with pytest.raises(ApiError) as caught:
        store.resolve(stale_request)
    assert caught.value.reason_code == "CONTEXT_PREVIEW_STALE"


def test_resolve_rejects_unavailable_operation_before_schema_validation(
    tmp_path: Path,
) -> None:
    _, _, store = _runtime(tmp_path)
    with pytest.raises(ApiError) as caught:
        store.resolve({"operation": "unavailable"})
    assert caught.value.code == "CAPABILITY_UNAVAILABLE"


def test_tiny_train_rejects_model_above_parameter_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, store = _runtime(tmp_path)
    dataset_id = str(uuid.uuid4())
    tokenizer_id = str(uuid.uuid4())
    monkeypatch.setattr(store, "_assert_request_references", lambda _value: None)
    monkeypatch.setattr(
        store,
        "_dataset",
        lambda *_args, **_kwargs: (
            {"manifest_sha256": "1" * 64},
            [
                {
                    "split": name,
                    "artifact_id": str(uuid.uuid4()),
                    "sealed": False,
                    "text_bytes": 1,
                }
                for name in ("train", "validation")
            ],
        ),
    )
    monkeypatch.setattr(
        store,
        "_tokenizer",
        lambda _tokenizer_id: (
            {"vocab_size": 65_536},
            {"role": "tokenizer.json"},
        ),
    )
    monkeypatch.setattr(store, "_require_tiny_dataset", lambda *_args: None)
    monkeypatch.setattr(store, "_enforce_text_cap", lambda *_args: None)

    with pytest.raises(ApiError) as caught:
        store.resolve(
            {
                "operation": "tiny_train",
                "dataset_id": dataset_id,
                "tokenizer_id": tokenizer_id,
                "architecture_profile_id": "tiny-v2-standard-v1",
                "steps": 1,
                "eval_every": 1,
                "batch_size": 1,
                "learning_rate": 0.001,
                "context": 512,
                "width": 512,
                "heads": 16,
                "layers": 12,
                "seed": 17,
            }
        )
    assert caught.value.reason_code == "SEMANTIC_INVALID"
    assert caught.value.field_errors[0]["field_path"] == "/width"
