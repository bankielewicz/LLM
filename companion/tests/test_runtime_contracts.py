from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from llm_foundations_companion.contract_data import ContractDataError, declared_files
from llm_foundations_companion.errors import ApiError
from llm_foundations_companion.limits import LIMITS
from llm_foundations_companion.schema import (
    SchemaDefinitionError,
    _Context,
    _assert_schema,
    _resolve_ref,
    canonical_json,
    load_document,
    strict_json,
    validate,
    validate_schema,
)


REPOSITORY = Path(__file__).resolve().parents[2]
SPEC_ROOT = REPOSITORY / "docs" / "specs" / "intermediate-v1"
FIXTURES = SPEC_ROOT / "fixtures" / "data"
REQUEST_ID = "123e4567-e89b-42d3-a456-426614174000"
ID_A = "123e4567-e89b-42d3-a456-426614174001"
ID_B = "123e4567-e89b-42d3-a456-426614174002"


def test_exact_runtime_limits_match_openapi_const_map() -> None:
    openapi = load_document("openapi.json")
    schema = openapi["components"]["schemas"]["RuntimeInfo"]
    properties = schema["properties"]["limits"]["properties"]
    expected = {name: rule["const"] for name, rule in properties.items()}
    assert dict(LIMITS) == expected
    assert tuple(LIMITS) == tuple(schema["properties"]["limits"]["required"])
    with pytest.raises(TypeError):
        LIMITS["queued_jobs"] = 9  # type: ignore[index]


def test_runtime_contract_builder_is_byte_exact() -> None:
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts" / "build_runtime_contracts.py"), "--check"],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "36 sealed documents" in completed.stdout


def test_materialized_fixture_metadata_is_bundled_without_fixture_text() -> None:
    source = (
        SPEC_ROOT
        / "fixtures"
        / "data"
        / "materialized"
        / "materialized-manifest.json"
    )
    packaged = (
        REPOSITORY
        / "companion"
        / "src"
        / "llm_foundations_companion"
        / "contract_data"
        / "materialized-manifest.json"
    )
    assert packaged.read_bytes() == source.read_bytes()
    assert "materialized-manifest.json" in declared_files()
    document = load_document("materialized-manifest.json")
    assert document["format"] == "llm-foundations-materialized-fixtures-v1"
    sealed = [
        row for row in document["files"]
        if row["path"] == "capstone-support-v1/sealed_test.jsonl"
    ]
    assert sealed == [
        {
            "path": "capstone-support-v1/sealed_test.jsonl",
            "records": 12,
            "sha256": "912c322aa0e866c5a068eb7a99b8fde7dce70703ba50b6294a231f74bb18d468",
            "utf8_bytes": 4643,
        }
    ]
    assert not any(path.suffix == ".jsonl" for path in packaged.parent.rglob("*"))


def test_load_document_is_defensive_and_aliases_standalone_schema() -> None:
    first = load_document("openapi.json")
    first["openapi"] = "tampered"
    assert load_document("openapi.json")["openapi"] != "tampered"
    assert load_document("common.schema.json") == load_document(
        "schemas/common.schema.json"
    )
    assert load_document("./schemas/dataset-manifest.schema.json") == load_document(
        "schemas/dataset-manifest.schema.json"
    )


@pytest.mark.parametrize(
    "name",
    (
        "../openapi.json",
        "../../openapi.json",
        "/schemas/common.schema.json",
        "C:/schemas/common.schema.json",
        "https://example.invalid/schema.json",
        "schemas\\common.schema.json",
        "schemas/common.schema.json?revision=1",
    ),
)
def test_load_document_rejects_names_outside_the_closed_bundle(name: str) -> None:
    with pytest.raises(ContractDataError):
        load_document(name)


@pytest.mark.parametrize(
    "raw,reason",
    [
        (b'{"a":1,"a":2}', "INVALID_JSON"),
        (b'{"a":NaN}', "INVALID_JSON"),
        (b'{"a":Infinity}', "INVALID_JSON"),
        (b'{"a":1e10000}', "INVALID_JSON"),
        (b'{"a":1} trailing', "INVALID_JSON"),
        (b'\xff', "INVALID_ENCODING"),
        ('"\ud800"', "INVALID_ENCODING"),
    ],
)
def test_strict_json_rejects_ambiguous_or_invalid_inputs(raw: bytes | str, reason: str) -> None:
    with pytest.raises(ApiError) as caught:
        strict_json(raw)
    assert caught.value.code == "VALIDATION_FAILED"
    assert caught.value.reason_code == reason
    assert caught.value.status_code == 400


def test_strict_and_canonical_json_preserve_contract_bytes() -> None:
    value = strict_json('{"é":"文字","z":1,"a":1.5}')
    assert value == {"é": "文字", "z": 1, "a": 1.5}
    assert canonical_json(value) == '{"a":1.5,"z":1,"é":"文字"}'.encode()
    assert not canonical_json(value).endswith(b"\n")
    decomposed = "e\N{COMBINING ACUTE ACCENT}"
    assert canonical_json({"text": decomposed}) == b'{"text":"e\xcc\x81"}'
    assert canonical_json({"text": decomposed}) != canonical_json({"text": "é"})
    with pytest.raises(ValueError):
        canonical_json({"bad": math.nan})
    with pytest.raises(ValueError):
        canonical_json({1: "not JSON"})  # type: ignore[dict-item]


def test_date_time_format_enforces_normative_utc_milliseconds() -> None:
    schema = {"type": "string", "format": "date-time"}
    assert validate_schema(schema, "2026-10-01T12:00:00.000Z") == (
        "2026-10-01T12:00:00.000Z"
    )
    for invalid in (
        "2026-10-01T12:00:00Z",
        "2026-10-01T12:00:00.00Z",
        "2026-10-01T12:00:00.0000Z",
        "2026-10-01T08:00:00.000-04:00",
    ):
        with pytest.raises(ApiError):
            validate_schema(schema, invalid)


def test_api_error_enforces_role_specific_bindings_and_envelope() -> None:
    error = ApiError(
        "VALIDATION_FAILED",
        reason_code="SCHEMA_INVALID",
        field_errors=[
            {"field_path": "/z", "message": "later"},
            {"field_path": "/a", "message": "earlier"},
        ],
    )
    assert error.status_code == error.http_status == 400
    assert error.retryable is False
    envelope = error.as_envelope(REQUEST_ID)
    assert envelope["error"]["reason_code"] == "SCHEMA_INVALID"
    assert [item["field_path"] for item in envelope["error"]["field_errors"]] == [
        "/a",
        "/z",
    ]
    assert validate("Error", envelope) == envelope

    disk = ApiError("DISK_FULL", reason_code="INSUFFICIENT_STORAGE")
    assert disk.status_code == 503
    assert disk.retryable is True
    with pytest.raises(ValueError):
        ApiError("VALIDATION_FAILED", reason_code="HOST_MISMATCH")
    with pytest.raises(ValueError):
        ApiError("PAYLOAD_TOO_LARGE", http_status=400)
    with pytest.raises(ValueError):
        error.as_error("NOT-A-UUID")


def _generate_request(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "operation": "generate",
        "checkpoint_id": ID_A,
        "prompt": "hello",
        "max_new_tokens": 8,
        "temperature": 1,
        "top_p": 1,
        "seed": 17,
        "preview_artifact_id": ID_B,
        "context_preview_digest": "a" * 64,
    }
    value.update(updates)
    return value


def test_openapi_component_validation_is_closed_and_type_normalizing() -> None:
    request = _generate_request()
    validated = validate("GenerateRequest", request)
    assert validated is not request
    assert validated["temperature"] == 1.0
    assert type(validated["temperature"]) is float
    assert type(validated["top_p"]) is float
    assert request["temperature"] == 1

    with pytest.raises(ApiError) as caught:
        validate("GenerateRequest", {**request, "unexpected": True})
    assert caught.value.reason_code == "SCHEMA_INVALID"
    assert caught.value.field_errors[0]["field_path"] == "/unexpected"


def test_x_reason_code_and_semantic_utf8_bounds_are_preserved() -> None:
    with pytest.raises(ApiError) as caught:
        validate("GenerateRequest", _generate_request(temperature=0, top_p=0.5))
    assert caught.value.reason_code == "INVALID_SAMPLING_PARAMETERS"
    assert caught.value.field_errors[0]["field_path"] == "/top_p"

    with pytest.raises(ApiError) as caught:
        validate("GenerateRequest", _generate_request(prompt="é" * 8_193))
    assert caught.value.reason_code == "PAYLOAD_TOO_LARGE"
    assert caught.value.field_errors[0]["field_path"] == "/prompt"


def test_tiny_request_pure_semantic_relations_run_before_acceptance() -> None:
    request = {
        "operation": "tiny_train",
        "dataset_id": ID_A,
        "tokenizer_id": ID_B,
        "architecture_profile_id": "tiny-v2-standard-v1",
        "steps": 20,
        "eval_every": 25,
        "batch_size": 8,
        "learning_rate": 0.0003,
        "seed": 17,
        "context": 64,
        "width": 65,
        "heads": 4,
        "layers": 2,
    }
    with pytest.raises(ApiError) as caught:
        validate("JobRequest", request)
    assert caught.value.code == "VALIDATION_FAILED"
    assert [item["field_path"] for item in caught.value.field_errors] == [
        "/eval_every",
        "/heads",
    ]


def test_inline_schema_refs_resolve_against_bundled_openapi() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "page"],
        "properties": {
            "id": {"$ref": "#/components/schemas/Identifier"},
            "page": {"type": "integer", "minimum": 1, "maximum": 100},
        },
    }
    value = {"id": ID_A, "page": 100}
    assert validate_schema(schema, value) == value
    with pytest.raises(ApiError):
        validate_schema(schema, {"id": ID_A, "page": 101})


def _all_references(value: object):
    if isinstance(value, dict):
        reference = value.get("$ref")
        if isinstance(reference, str):
            yield reference
        for child in value.values():
            yield from _all_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_references(child)


def test_every_bundled_reference_resolves_from_its_owning_document() -> None:
    resolved: list[tuple[str, str, str]] = []
    for owner in declared_files():
        document = load_document(owner)
        context = _Context(document=owner, root=document)
        for reference in _all_references(document):
            target, target_context = _resolve_ref(reference, context)
            _assert_schema(target)
            resolved.append((owner, reference, target_context.document))

    assert len(resolved) == 683
    assert (
        "openapi.json",
        "./schemas/dataset-manifest.schema.json",
        "schemas/dataset-manifest.schema.json",
    ) in resolved
    assert (
        "schemas/dataset-manifest.schema.json",
        "common.schema.json#/$defs/id",
        "schemas/common.schema.json",
    ) in resolved


def test_openapi_component_can_join_two_external_schema_documents() -> None:
    value = {
        "format": "llm-foundations-dataset-v1",
        "dataset_id": ID_A,
        "name": "Native direct registration",
        "record_format": "document_text_v1",
        "origin": "imported",
        "splits": {
            "train": {
                "artifact_id": ID_B,
                "sha256": "a" * 64,
                "records": 1,
                "utf8_bytes": 3,
                "sealed": False,
            }
        },
        "audit_artifact_id": "123e4567-e89b-42d3-a456-426614174003",
        "created_at": "2026-10-01T12:00:00.000Z",
        "manifest_sha256": "b" * 64,
        "eligibility": "audit_only",
    }
    assert validate("DatasetSummary", value) == value


def test_unsupported_validation_keywords_fail_closed() -> None:
    with pytest.raises(SchemaDefinitionError, match="unsupported JSON Schema keyword"):
        validate_schema({"type": "string", "madeUpMinimum": 3}, "abc")


def _walk_schema_definitions(schema: object):
    assert isinstance(schema, (dict, bool))
    _assert_schema(schema)
    yield schema
    if isinstance(schema, bool):
        return
    for keyword in ("$defs", "properties"):
        children = schema.get(keyword, {})
        assert isinstance(children, dict)
        for child in children.values():
            yield from _walk_schema_definitions(child)
    for keyword in (
        "additionalProperties",
        "items",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
    ):
        if keyword in schema:
            yield from _walk_schema_definitions(schema[keyword])
    for keyword in ("prefixItems", "allOf", "anyOf", "oneOf"):
        children = schema.get(keyword, [])
        assert isinstance(children, list)
        for child in children:
            yield from _walk_schema_definitions(child)


def _content_schemas(owner: object):
    assert isinstance(owner, dict)
    content = owner.get("content", {})
    assert isinstance(content, dict)
    for media in content.values():
        assert isinstance(media, dict)
        if "schema" in media:
            yield media["schema"]


def test_every_frozen_openapi_parameter_body_and_response_schema_is_supported() -> None:
    openapi = load_document("openapi.json")
    roots: list[object] = list(openapi["components"]["schemas"].values())
    operation_count = 0
    method_names = {"get", "post", "put", "patch", "delete", "options", "head"}

    for path_item in openapi["paths"].values():
        for parameter in path_item.get("parameters", []):
            if "schema" in parameter:
                roots.append(parameter["schema"])
        for method, operation in path_item.items():
            if method not in method_names:
                continue
            operation_count += 1
            for parameter in operation.get("parameters", []):
                if "schema" in parameter:
                    roots.append(parameter["schema"])
            roots.extend(_content_schemas(operation.get("requestBody", {})))
            for response in operation["responses"].values():
                roots.extend(_content_schemas(response))
                for header in response.get("headers", {}).values():
                    if "schema" in header:
                        roots.append(header["schema"])

    for name in declared_files():
        if name.startswith("schemas/"):
            roots.append(load_document(name))

    assert operation_count == 57
    units = [unit for root in roots for unit in _walk_schema_definitions(root)]
    assert len(units) >= 500


def test_stdlib_evaluator_matches_jsonschema_oracle_for_supported_keywords() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind", "values", "label"],
        "properties": {
            "kind": {"enum": ["a", "b"]},
            "values": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "uniqueItems": True,
                "contains": {"type": "integer", "minimum": 2},
                "minContains": 1,
                "items": {"type": "integer", "minimum": 0, "maximum": 5},
            },
            "label": {"type": "string", "minLength": 1, "maxLength": 5, "pattern": "^[a-z]+$"},
            "paired": {"type": "string"},
            "peer": {"type": "string"},
        },
        "dependentRequired": {"paired": ["peer"]},
        "allOf": [
            {
                "if": {"properties": {"kind": {"const": "b"}}, "required": ["kind"]},
                "then": {"properties": {"values": {"minItems": 2}}},
            }
        ],
    }
    values = [
        {"kind": "a", "values": [2], "label": "ok"},
        {"kind": "b", "values": [1, 2], "label": "good", "paired": "x", "peer": "y"},
        {"kind": "b", "values": [2], "label": "ok"},
        {"kind": "a", "values": [1], "label": "ok"},
        {"kind": "a", "values": [2, 2], "label": "ok"},
        {"kind": "a", "values": [2], "label": "TOO-LONG"},
        {"kind": "a", "values": [2], "label": "ok", "paired": "x"},
        {"kind": "a", "values": [2], "label": "ok", "extra": 1},
    ]
    oracle = Draft202012Validator(schema)
    for value in values:
        expected = oracle.is_valid(value)
        try:
            validate_schema(schema, value)
            actual = True
        except ApiError:
            actual = False
        assert actual is expected, value


def test_all_frozen_schema_fixture_expectations_match_runtime_validator() -> None:
    case_document = strict_json((FIXTURES / "validation-cases.json").read_bytes())
    assert isinstance(case_document, dict)
    cases = case_document["cases"]
    observed: list[tuple[str, bool]] = []
    for case in cases:
        schema_name = Path(case["schema"]).name
        instance = strict_json((FIXTURES / case["instance"]).read_bytes())
        try:
            validate(schema_name, instance)
            valid = True
        except ApiError:
            valid = False
        observed.append((case["id"], valid))
        assert valid is case["valid"], case["id"]
    assert len(observed) == 49
    assert len({case_id for case_id, _ in observed}) == 49
