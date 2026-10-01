from __future__ import annotations

import io
import json
import os
import uuid
from pathlib import Path

import pytest

from llm_foundations_companion.database import Database
from llm_foundations_companion.datasets import DatasetRegistry
from llm_foundations_companion.errors import ApiError
from llm_foundations_companion.registry import Registry
from llm_foundations_companion.schema import canonical_json


REPOSITORY = Path(__file__).resolve().parents[2]
MATERIALIZED = (
    REPOSITORY / "docs" / "specs" / "intermediate-v1" / "fixtures" / "data" / "materialized"
)


def _stack(tmp_path: Path) -> tuple[Database, Registry, DatasetRegistry]:
    root = tmp_path / "root"
    database = Database(root)
    database.initialize()
    registry = Registry(database, root, str(uuid.uuid4()), os.urandom(32))
    return database, registry, DatasetRegistry(registry)


def _reserve(registry: Registry) -> str:
    owner = str(uuid.uuid4())
    registry.reserve_capacity(
        owner_kind="upload",
        owner_id=owner,
        byte_count=64 * 1024 * 1024,
        artifact_rows=5,
        dataset_rows=1,
        reservation_id=owner,
    )
    return owner


def _register_family(
    datasets: DatasetRegistry,
    registry: Registry,
    family: str,
    record_format: str,
    source_names: tuple[str, ...],
) -> dict[str, object]:
    streams = {
        ("test" if name == "sealed_test" else "train" if name == "documents" else name): (
            MATERIALIZED / family / f"{name}.jsonl"
        ).open("rb")
        for name in source_names
    }
    try:
        return datasets.register(
            {"name": family, "record_format": record_format},
            streams,
            upload_id=_reserve(registry),
        )
    finally:
        for stream in streams.values():
            stream.close()


@pytest.mark.parametrize(
    "family,record_format,names,expected",
    [
        (
            "data-clinic-v1",
            "document_text_v1",
            ("train", "validation", "test"),
            (48, 0, 0, 0, "eligible"),
        ),
        (
            "data-clinic-leaky-v1",
            "document_text_v1",
            ("train", "validation", "test"),
            (48, 1, 2, 1, "audit_only"),
        ),
        (
            "applied-intents-v1",
            "instruction_intent_v1",
            ("train", "validation", "test"),
            (84, 0, 0, 0, "eligible"),
        ),
        (
            "capstone-support-v1",
            "instruction_intent_v1",
            ("train", "validation", "sealed_test"),
            (48, 0, 0, 0, "eligible"),
        ),
        (
            "retrieval-manual-v1",
            "retrieval_v1",
            ("documents",),
            (8, 0, 0, 0, "eligible"),
        ),
    ],
)
def test_authoritative_fixture_registration_matches_audit_oracles(
    tmp_path: Path,
    family: str,
    record_format: str,
    names: tuple[str, ...],
    expected: tuple[int, int, int, int, str],
) -> None:
    _, registry, datasets = _stack(tmp_path)
    manifest = _register_family(datasets, registry, family, record_format, names)
    audit = json.loads(registry.read_artifact_bytes(str(manifest["audit_artifact_id"])))
    actual = (
        audit["record_count"],
        audit["exact_duplicate_pairs"],
        audit["normalized_near_duplicate_pairs"],
        audit["group_overlap_count"],
        audit["eligibility"],
    )
    assert actual == expected
    assert manifest["origin"] == "shipped_fixture"
    assert datasets.get(str(manifest["dataset_id"])) == manifest


def test_content_only_manifest_digest_ignores_name_time_and_local_ids(
    tmp_path: Path,
) -> None:
    _, registry, datasets = _stack(tmp_path)
    raw = b'{"record_id":"r1","scenario_group_id":"g1","text":"same"}\n'
    first = datasets.register(
        {"name": "first", "record_format": "document_text_v1"},
        {"train": io.BytesIO(raw)},
        upload_id=_reserve(registry),
    )
    second = datasets.register(
        {
            "name": "second",
            "record_format": "document_text_v1",
            "provenance_note": "different local provenance",
        },
        {"train": io.BytesIO(raw)},
        upload_id=_reserve(registry),
    )
    assert first["dataset_id"] != second["dataset_id"]
    assert first["splits"]["train"]["artifact_id"] != second["splits"]["train"]["artifact_id"]
    assert first["manifest_sha256"] == second["manifest_sha256"]


def test_partial_authoritative_family_does_not_claim_shipped_provenance(
    tmp_path: Path,
) -> None:
    _, registry, datasets = _stack(tmp_path)
    manifest = _register_family(
        datasets,
        registry,
        "data-clinic-v1",
        "document_text_v1",
        ("train",),
    )
    assert manifest["origin"] == "imported"


def test_invalid_jsonl_and_duplicate_ids_leave_registry_revision_unchanged(
    tmp_path: Path,
) -> None:
    database, registry, datasets = _stack(tmp_path)
    before = database.revision
    owner = _reserve(registry)
    with pytest.raises(ApiError) as malformed:
        datasets.register(
            {"name": "bad", "record_format": "document_text_v1"},
            {"train": io.BytesIO(b'{"record_id":\n')},
            upload_id=owner,
        )
    assert malformed.value.reason_code == "INVALID_JSON"
    assert database.revision == before
    registry.release_capacity_now(owner)
    duplicate = (
        b'{"record_id":"same","scenario_group_id":"g1","text":"one"}\n'
        b'{"record_id":"same","scenario_group_id":"g2","text":"two"}\n'
    )
    owner = _reserve(registry)
    with pytest.raises(ApiError) as repeated:
        datasets.register(
            {"name": "duplicate", "record_format": "document_text_v1"},
            {"train": io.BytesIO(duplicate)},
            upload_id=owner,
        )
    assert repeated.value.reason_code == "DUPLICATE_SOURCE"
    assert database.revision == before


def test_whitespace_text_is_a_line_specific_semantic_failure(tmp_path: Path) -> None:
    _, registry, datasets = _stack(tmp_path)
    rows = [
        {
            "record_id": f"r{index}",
            "scenario_group_id": f"g{index}",
            "text": "   " if index == 17 else "valid",
        }
        for index in range(1, 18)
    ]
    raw = b"\n".join(canonical_json(row) for row in rows) + b"\n"
    with pytest.raises(ApiError) as caught:
        datasets.register(
            {"name": "semantic", "record_format": "document_text_v1"},
            {"train": io.BytesIO(raw)},
            upload_id=_reserve(registry),
        )
    assert caught.value.reason_code == "SEMANTIC_INVALID"
    assert [dict(item) for item in caught.value.field_errors] == [
        {"field_path": "/train/17/text", "message": "Text must be nonblank and contain no NUL."}
    ]


def test_jsonl_does_not_treat_unicode_line_separator_as_record_boundary(
    tmp_path: Path,
) -> None:
    _, registry, datasets = _stack(tmp_path)
    raw = canonical_json(
        {"record_id": "r1", "scenario_group_id": "g1", "text": "left\u2028right"}
    ) + b"\n"
    manifest = datasets.register(
        {"name": "unicode separator", "record_format": "document_text_v1"},
        {"train": io.BytesIO(raw)},
        upload_id=_reserve(registry),
    )
    assert manifest["splits"]["train"]["records"] == 1


def test_cross_reference_stage_precedes_semantics_across_splits(
    tmp_path: Path,
) -> None:
    _, registry, datasets = _stack(tmp_path)
    train = b'{"record_id":"same","scenario_group_id":"g1","text":"   "}\n'
    test = b'{"record_id":"same","scenario_group_id":"g2","text":"valid"}\n'
    with pytest.raises(ApiError) as caught:
        datasets.register(
            {"name": "stage order", "record_format": "document_text_v1"},
            {"train": io.BytesIO(train), "test": io.BytesIO(test)},
            upload_id=_reserve(registry),
        )
    assert caught.value.reason_code == "DUPLICATE_SOURCE"
    assert [item["field_path"] for item in caught.value.field_errors] == [
        "/test/1/record_id"
    ]


def test_retrieval_rejects_nontrain_part_before_registration(tmp_path: Path) -> None:
    database, registry, datasets = _stack(tmp_path)
    before = database.revision
    with pytest.raises(ApiError) as caught:
        datasets.register(
            {"name": "retrieval", "record_format": "retrieval_v1"},
            {"validation": io.BytesIO(b'{"record_id":"r","text":"text"}\n')},
            upload_id=_reserve(registry),
        )
    assert caught.value.reason_code == "SEMANTIC_INVALID"
    assert database.revision == before


def test_sealed_digest_is_never_exposed_by_ordinary_artifact_content(
    tmp_path: Path,
) -> None:
    _, registry, datasets = _stack(tmp_path)
    manifest = _register_family(
        datasets,
        registry,
        "capstone-support-v1",
        "instruction_intent_v1",
        ("train", "validation", "sealed_test"),
    )
    test_split = manifest["splits"]["test"]
    assert test_split["sealed"] is True
    artifact_id = test_split["artifact_id"]
    with pytest.raises(ApiError) as caught:
        registry.read_artifact_bytes(artifact_id)
    assert (caught.value.code, caught.value.reason_code) == (
        "STATE_CONFLICT",
        "SEALED_SPLIT_REQUIRES_TOKEN",
    )
    trusted = registry.read_artifact_bytes(artifact_id, allow_sealed_internal=True)
    assert len(trusted) == test_split["utf8_bytes"]


def test_manifest_and_audit_are_exact_frozen_schemas(tmp_path: Path) -> None:
    _, registry, datasets = _stack(tmp_path)
    raw = b'{"record_id":"r1","scenario_group_id":"g1","text":"hello","slice":"demo"}\n'
    manifest = datasets.register(
        {"name": "schema check", "record_format": "document_text_v1"},
        {"train": io.BytesIO(raw)},
        upload_id=_reserve(registry),
    )
    audit = json.loads(registry.read_artifact_bytes(manifest["audit_artifact_id"]))
    assert audit["split_counts"] == {"train": 1}
    assert audit["slice_counts"] == {"demo": 1}
    assert audit["utf8_bytes"] == len("hello".encode())
    identity = {
        "record_format": manifest["record_format"],
        "splits": {
            key: {
                "sha256": value["sha256"],
                "records": value["records"],
                "utf8_bytes": value["utf8_bytes"],
                "sealed": value["sealed"],
            }
            for key, value in manifest["splits"].items()
        },
    }
    import hashlib

    assert manifest["manifest_sha256"] == hashlib.sha256(canonical_json(identity)).hexdigest()
