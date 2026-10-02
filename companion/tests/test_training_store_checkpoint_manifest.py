from __future__ import annotations

import hashlib
import io
import json
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from llm_foundations_companion.database import Database
from llm_foundations_companion.errors import ApiError
from llm_foundations_companion.registry import Registry
from llm_foundations_companion.schema import canonical_json
from llm_foundations_companion.training_store import TrainingStore


def _checkpoint_runtime(
    tmp_path: Path,
    *,
    manifest_change: Callable[[dict[str, Any]], bytes] | None = None,
    omitted_file: str | None = None,
    extra_file: bool = False,
) -> tuple[TrainingStore, Registry, dict[str, dict[str, Any]]]:
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

    payloads = {
        "model.safetensors": b"model-bytes",
        "tokenizer.json": b'{"format":"teaching-byte-bpe-v1","merges":[]}',
        "config.json": b'{"format":"tiny-v2-config-v1"}',
    }
    manifest_value: dict[str, Any] = {
        "format": "tiny-v2-checkpoint-v1",
        "portability": "inference_only",
        "files": [
            {
                "name": name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, payload in payloads.items()
        ],
        "aliases": {},
    }
    manifest_raw = (
        canonical_json(manifest_value)
        if manifest_change is None
        else manifest_change(manifest_value)
    )
    payloads["manifest.json"] = manifest_raw
    if extra_file:
        payloads["metrics.jsonl"] = b"extra"

    job_id = str(uuid.uuid4())
    artifacts = {
        name: registry.register_stream(
            io.BytesIO(payload),
            job_id,
            artifact_type="checkpoint_file",
            display_name=name,
            media_type=(
                "application/json"
                if name.endswith(".json")
                else "application/octet-stream"
            ),
            preview_policy="metadata_only",
            origin="locally_created",
        )
        for name, payload in payloads.items()
    }
    checkpoint_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    dataset_id = str(uuid.uuid4())
    now = "2026-10-01T00:00:00.000Z"
    descriptor = {
        "checkpoint_id": checkpoint_id,
        "artifact_id": artifacts["manifest.json"]["artifact_id"],
        "sha256": artifacts["manifest.json"]["sha256"],
        "format": "tiny_v2_portable",
        "origin": "locally_created",
        "run_id": run_id,
        "job_id": job_id,
        "backend": "tiny_v2",
        "step": 0,
        "model_identity": {
            "kind": "tiny_v2",
            "architecture_profile_id": "tiny-v2-standard-v1",
            "config_sha256": artifacts["config.json"]["sha256"],
            "tokenizer_sha256": artifacts["tokenizer.json"]["sha256"],
        },
        "dataset_bindings": [
            {
                "dataset_id": dataset_id,
                "split": split,
                "artifact_id": str(uuid.uuid4()),
                "sha256": digest * 64,
            }
            for split, digest in (("train", "3"), ("validation", "4"))
        ],
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
        "seed": 7,
        "dependency_lock_sha256": "1" * 64,
        "dtype": "float32",
        "runtime_profile": "wsl-cpu",
        "created_at": now,
        "portability": "inference_only",
        "contains_resume_state": False,
        "parent_checkpoint_id": None,
        "device": "cpu",
        "companion_source_revision": "2" * 40,
    }
    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO runs(
                run_id, format, operation, origin, job_id, source_identity_json,
                created_at, updated_at, record_json
            ) VALUES (?, 'tiny-v2-run-v1', 'tiny_train', 'locally_created',
                      ?, NULL, ?, ?, '{}')
            """,
            (run_id, job_id, now, now),
        )
        connection.execute(
            """
            INSERT INTO checkpoints(
                checkpoint_id, format, backend, origin, run_id, job_id,
                manifest_sha256, source_identity_json, created_at, updated_at,
                record_json
            ) VALUES (?, 'tiny_v2_portable', 'tiny_v2', 'locally_created',
                      ?, ?, ?, NULL, ?, ?, ?)
            """,
            (
                checkpoint_id,
                run_id,
                job_id,
                artifacts["manifest.json"]["sha256"],
                now,
                now,
                json.dumps(descriptor, separators=(",", ":"), sort_keys=True),
            ),
        )
        for name, artifact in artifacts.items():
            if name == omitted_file:
                continue
            connection.execute(
                """
                INSERT INTO checkpoint_files(
                    checkpoint_id, file_name, artifact_id, sha256, size_bytes
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    name,
                    artifact["artifact_id"],
                    artifact["sha256"],
                    artifact["size_bytes"],
                ),
            )
    artifacts["_checkpoint"] = {"checkpoint_id": checkpoint_id}
    return store, registry, artifacts


def _checkpoint_id(artifacts: dict[str, dict[str, Any]]) -> str:
    return str(artifacts["_checkpoint"]["checkpoint_id"])


def _assert_checkpoint_incompatible(exc: ApiError) -> None:
    assert exc.status_code == 400
    assert exc.code == "VALIDATION_FAILED"
    assert exc.reason_code == "CHECKPOINT_INCOMPATIBLE"
    assert [item["field_path"] for item in exc.field_errors] == ["/checkpoint_id"]


def test_checkpoint_accepts_manifest_bound_exact_file_set(tmp_path: Path) -> None:
    store, _, artifacts = _checkpoint_runtime(tmp_path)

    checkpoint, inputs = store._checkpoint(_checkpoint_id(artifacts))

    assert checkpoint["portability"] == "inference_only"
    assert set(checkpoint["_files"]) == {
        "config.json",
        "manifest.json",
        "model.safetensors",
        "tokenizer.json",
    }
    assert {item["role"] for item in inputs} == {
        "checkpoint.config.json",
        "checkpoint.manifest.json",
        "checkpoint.model.safetensors",
        "checkpoint.tokenizer.json",
    }


@pytest.mark.parametrize("case", ["malformed", "digest", "aliases"])
def test_checkpoint_rejects_valid_bytes_with_incompatible_manifest_metadata(
    tmp_path: Path, case: str
) -> None:
    def change(value: dict[str, Any]) -> bytes:
        if case == "malformed":
            return b'{"format":'
        if case == "digest":
            value["files"][0]["sha256"] = "f" * 64
        else:
            value["aliases"] = {"output.weight": "tokens.weight"}
        return canonical_json(value)

    store, _, artifacts = _checkpoint_runtime(tmp_path, manifest_change=change)

    with pytest.raises(ApiError) as caught:
        store._checkpoint(_checkpoint_id(artifacts))

    _assert_checkpoint_incompatible(caught.value)


@pytest.mark.parametrize(
    ("omitted_file", "extra_file"),
    [("model.safetensors", False), (None, True)],
)
def test_checkpoint_rejects_database_file_set_that_differs_from_manifest(
    tmp_path: Path, omitted_file: str | None, extra_file: bool
) -> None:
    store, _, artifacts = _checkpoint_runtime(
        tmp_path, omitted_file=omitted_file, extra_file=extra_file
    )

    with pytest.raises(ApiError) as caught:
        store._checkpoint(_checkpoint_id(artifacts))

    _assert_checkpoint_incompatible(caught.value)


@pytest.mark.parametrize("corrupt_name", ["manifest.json", "model.safetensors"])
def test_checkpoint_physical_corruption_wins_over_manifest_semantics(
    tmp_path: Path, corrupt_name: str
) -> None:
    def incompatible_alias(value: dict[str, Any]) -> bytes:
        value["aliases"] = {"output.weight": "tokens.weight"}
        return canonical_json(value)

    store, registry, artifacts = _checkpoint_runtime(
        tmp_path, manifest_change=incompatible_alias
    )
    artifact = artifacts[corrupt_name]
    path = registry.root.joinpath(*str(artifact["relative_path"]).split("/"))
    path.write_bytes(path.read_bytes() + b"corrupt")

    with pytest.raises(ApiError) as caught:
        store._checkpoint(_checkpoint_id(artifacts))

    assert caught.value.code == "STORAGE_UNAVAILABLE"
    assert caught.value.reason_code == "STORAGE_CORRUPT"
