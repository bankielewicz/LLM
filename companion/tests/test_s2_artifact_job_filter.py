from __future__ import annotations

import os
import uuid
from pathlib import Path
from urllib.parse import quote

import pytest

from llm_foundations_companion.database import Database
from llm_foundations_companion.registry import Registry
from llm_foundations_companion.service import Service
from test_api import Harness


def _runtime(tmp_path: Path) -> tuple[Registry, Service, Harness]:
    root = tmp_path / "root"
    database = Database(root)
    database.initialize()
    registry = Registry(database, root, str(uuid.uuid4()), os.urandom(32))
    service = object.__new__(Service)
    service.registry = registry
    harness = Harness()
    harness.list_artifacts = service.list_artifacts
    return registry, service, harness


def _artifact(
    registry: Registry,
    *,
    job_id: str | None,
    artifact_type: str,
    origin: str,
    label: str,
) -> dict[str, object]:
    return registry.register_stream(
        label.encode(),
        str(uuid.uuid4()),
        artifact_type=artifact_type,
        display_name=f"{label}.json",
        media_type="application/json",
        preview_policy="metadata_only",
        origin=origin,
        job_id=job_id,
    )


def _query(**values: object) -> bytes:
    return "&".join(
        f"{name}={quote(str(value), safe='')}" for name, value in values.items()
    ).encode()


def test_service_adapter_preserves_all_registry_filters(tmp_path: Path) -> None:
    registry, service, _ = _runtime(tmp_path)
    selected_job = str(uuid.uuid4())
    other_job = str(uuid.uuid4())
    wanted = _artifact(
        registry,
        job_id=selected_job,
        artifact_type="generation_output",
        origin="imported",
        label="wanted",
    )
    _artifact(
        registry,
        job_id=selected_job,
        artifact_type="generation_output",
        origin="locally_created",
        label="wrong-origin",
    )
    _artifact(
        registry,
        job_id=selected_job,
        artifact_type="metrics",
        origin="imported",
        label="wrong-type",
    )
    _artifact(
        registry,
        job_id=other_job,
        artifact_type="generation_output",
        origin="imported",
        label="wrong-job",
    )

    page = service.list_artifacts(
        {
            "job_id": selected_job,
            "type": "generation_output",
            "origin": "imported",
            "limit": 100,
        }
    )

    assert [item["artifact_id"] for item in page["items"]] == [
        wanted["artifact_id"]
    ]
    assert page["next_cursor"] is None


def test_api_job_and_type_filter_pagination_is_complete_and_cursor_bound(
    tmp_path: Path,
) -> None:
    registry, service, harness = _runtime(tmp_path)
    selected_job = str(uuid.uuid4())
    other_job = str(uuid.uuid4())
    matching = {
        str(
            _artifact(
                registry,
                job_id=selected_job,
                artifact_type="generation_output",
                origin=origin,
                label=f"match-{index}",
            )["artifact_id"]
        )
        for index, origin in enumerate(
            ("imported", "locally_created", "legacy_imported")
        )
    }
    _artifact(
        registry,
        job_id=selected_job,
        artifact_type="metrics",
        origin="locally_created",
        label="wrong-type",
    )
    _artifact(
        registry,
        job_id=other_job,
        artifact_type="generation_output",
        origin="locally_created",
        label="wrong-job",
    )
    _artifact(
        registry,
        job_id=None,
        artifact_type="generation_output",
        origin="imported",
        label="no-job",
    )

    complete = service.list_artifacts(
        {"job_id": selected_job, "type": "generation_output", "limit": 100}
    )
    assert {str(item["artifact_id"]) for item in complete["items"]} == matching

    status, first, _ = harness.request(
        "/api/v1/artifacts",
        query=_query(
            job_id=selected_job,
            type="generation_output",
            limit=2,
        ),
    )
    assert status == 200, first
    assert len(first["items"]) == 2
    assert first["next_cursor"]
    assert len(first["next_cursor"]) <= 200

    status, second, _ = harness.request(
        "/api/v1/artifacts",
        query=_query(
            job_id=selected_job,
            type="generation_output",
            limit=2,
            cursor=first["next_cursor"],
        ),
    )
    assert status == 200
    assert len(second["items"]) == 1
    assert second["next_cursor"] is None
    paged = {
        str(item["artifact_id"]) for item in first["items"] + second["items"]
    }
    assert paged == matching

    for changed in (
        {"job_id": other_job, "type": "generation_output"},
        {"job_id": selected_job, "type": "metrics"},
        {"type": "generation_output"},
    ):
        status, body, _ = harness.request(
            "/api/v1/artifacts",
            query=_query(limit=2, cursor=first["next_cursor"], **changed),
        )
        assert status == 400
        assert body["error"]["reason_code"] == "SEMANTIC_INVALID"


def test_api_job_filter_validates_identifier_and_unknown_job_is_empty(
    tmp_path: Path,
) -> None:
    _, _, harness = _runtime(tmp_path)

    status, body, _ = harness.request(
        "/api/v1/artifacts", query=b"job_id=not-a-uuid&limit=100"
    )
    assert status == 400
    assert body["error"]["reason_code"] == "SCHEMA_INVALID"
    assert body["error"]["field_errors"][0]["field_path"] == "job_id"

    status, body, _ = harness.request(
        "/api/v1/artifacts",
        query=_query(job_id=str(uuid.uuid4()), limit=100),
    )
    assert status == 200
    assert body == {"items": [], "next_cursor": None}

    status, body, _ = harness.request(
        "/api/v1/artifacts", query=b"origin=imported"
    )
    assert status == 400
    assert body["error"]["reason_code"] == "SCHEMA_INVALID"
    assert body["error"]["field_errors"][0]["field_path"] == "origin"
