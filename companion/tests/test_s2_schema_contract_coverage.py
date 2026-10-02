from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import llm_foundations_companion.contracts as contracts_module
from llm_foundations_companion.contract_data import ContractDataError
from llm_foundations_companion.contracts import (
    ContractCatalog,
    ContractSourceError,
    error_bindings,
    strict_json_load,
    validate_fixture_cases,
    validate_model_profile_fixture,
    validate_openapi_shape,
    validate_reference_closure,
    validate_schema_units,
    validate_semantic_and_error_bindings,
)
from llm_foundations_companion.errors import ApiError
from llm_foundations_companion.limits import LIMITS
from llm_foundations_companion.schema import (
    SchemaDefinitionError,
    strict_json,
    validate,
    validate_schema,
)


REPOSITORY = Path(__file__).resolve().parents[2]
SPEC_ROOT = REPOSITORY / "docs" / "specs" / "intermediate-v1"
MATRIX_PATH = SPEC_ROOT / "fixtures" / "data" / "validation-cases.json"
PROFILE_PATH = SPEC_ROOT / "fixtures" / "applied" / "smollm2-135m-instruct-v1.json"
MANIFEST_PATH = SPEC_ROOT / "fixtures" / "applied" / "model-download-manifest.json"


@pytest.fixture(scope="module")
def catalog() -> ContractCatalog:
    return ContractCatalog.from_spec_root(SPEC_ROOT)


def _replace_document(catalog: ContractCatalog, path: Path, document: Any) -> ContractCatalog:
    documents = dict(catalog.documents)
    documents[path.resolve()] = document
    return replace(catalog, documents=documents)


def _issue_codes(issues: tuple[Any, ...] | list[Any]) -> set[str]:
    return {issue.code for issue in issues}


def test_contract_json_file_failures_are_typed(tmp_path: Path) -> None:
    with pytest.raises(ContractSourceError) as missing:
        strict_json_load(tmp_path / "missing.json")
    assert missing.value.code == "SOURCE_UNREADABLE"

    encoded = tmp_path / "encoded.json"
    encoded.write_bytes(b"\xff")
    with pytest.raises(ContractSourceError) as invalid_encoding:
        strict_json_load(encoded)
    assert invalid_encoding.value.code == "JSON_ENCODING_INVALID"

    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"open":', encoding="utf-8")
    with pytest.raises(ContractSourceError) as invalid_json:
        strict_json_load(malformed)
    assert invalid_json.value.code == "JSON_INVALID"
    assert invalid_json.value.source == malformed.resolve()


def test_catalog_entry_points_reject_missing_and_unregistered_contracts(
    catalog: ContractCatalog,
    tmp_path: Path,
) -> None:
    with pytest.raises(ContractSourceError) as missing_root:
        ContractCatalog.from_spec_root(tmp_path / "absent")
    assert missing_root.value.code == "CONTRACT_ROOT_MISSING"

    contracts = tmp_path / "partial" / "contracts"
    contracts.mkdir(parents=True)
    (contracts / "model-profile.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ContractSourceError) as missing_source:
        ContractCatalog.from_spec_root(contracts.parent)
    assert missing_source.value.code == "CONTRACT_SOURCE_MISSING"

    outside = tmp_path / "outside.schema.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(ContractSourceError) as unregistered:
        catalog.schema_errors(outside, {})
    assert unregistered.value.code == "SCHEMA_NOT_REGISTERED"

    with pytest.raises(ContractSourceError) as missing_component:
        catalog.component_errors("NoSuchComponent", {})
    assert missing_component.value.code == "OPENAPI_COMPONENT_MISSING"

    non_object = _replace_document(catalog, catalog.openapi_path, [])
    with pytest.raises(ContractSourceError) as invalid_openapi:
        _ = non_object.openapi
    assert invalid_openapi.value.code == "OPENAPI_TYPE_INVALID"


def test_reference_closure_reports_closed_path_and_fragment_failures(tmp_path: Path) -> None:
    contract_root = tmp_path / "contracts"
    contract_root.mkdir()
    owner = contract_root / "owner.json"
    owner.write_text("{}", encoding="utf-8")
    document = {
        "scalar": 1,
        "bad_type": {"$ref": 7},
        "absolute": {"$ref": "/outside.json"},
        "escape": {"$ref": "../outside.json"},
        "query": {"$ref": "local.json?revision=1"},
        "anchor": {"$ref": "#named-anchor"},
        "through_scalar": {"$ref": "#/scalar/child"},
    }

    count, issues = validate_reference_closure(contract_root, {owner: document})

    assert count == 6
    assert _issue_codes(issues) == {
        "REFERENCE_TYPE_INVALID",
        "REFERENCE_PATH_INVALID",
        "REFERENCE_ESCAPE",
        "REFERENCE_NETWORK_FORBIDDEN",
        "REFERENCE_FRAGMENT_UNSUPPORTED",
        "REFERENCE_UNRESOLVED",
    }


def test_schema_unit_validation_reports_invalid_standalone_component_and_path_schema(
    catalog: ContractCatalog,
) -> None:
    invalid = {"type": "definitely-not-a-json-schema-type"}
    documents = dict(catalog.documents)
    documents[catalog.schema_paths[0]] = invalid
    openapi = copy.deepcopy(catalog.openapi)
    openapi["components"]["schemas"]["BrokenForCoverage"] = invalid
    first_path = next(iter(openapi["paths"].values()))
    first_operation = next(
        value for key, value in first_path.items() if key in {"get", "post", "put", "patch", "delete"}
    )
    first_operation.setdefault("parameters", []).append(
        {"in": "query", "name": "broken", "schema": invalid}
    )
    documents[catalog.openapi_path] = openapi

    count, issues = validate_schema_units(replace(catalog, documents=documents))

    assert count > 0
    invalid_issues = [issue for issue in issues if issue.code == "SCHEMA_INVALID"]
    assert len(invalid_issues) >= 3
    assert any(issue.location == "" for issue in invalid_issues)
    assert any("BrokenForCoverage" in issue.location for issue in invalid_issues)
    assert any(issue.location.startswith("/paths") for issue in invalid_issues)


def test_openapi_shape_reports_each_structural_failure(catalog: ContractCatalog) -> None:
    malformed = {
        "openapi": "3.0.0",
        "info": {"title": ""},
        "servers": [{"url": "http://0.0.0.0:8765"}],
        "paths": {
            "relative": 1,
            "/not-an-operation": {"get": None},
            "/missing": {"post": {"responses": {}}},
            "/first": {"get": {"operationId": "same", "responses": {"200": {}}}},
            "/second": {"post": {"operationId": "same", "responses": {"200": {}}}},
        },
        "components": {"schemas": {}},
    }
    mutated = _replace_document(catalog, catalog.openapi_path, malformed)

    operation_count, issues = validate_openapi_shape(mutated)

    assert operation_count == 4
    assert _issue_codes(issues) == {
        "OPENAPI_VERSION_INVALID",
        "OPENAPI_INFO_INVALID",
        "OPENAPI_SERVER_INVALID",
        "OPENAPI_PATH_INVALID",
        "OPENAPI_PATH_ITEM_INVALID",
        "OPENAPI_OPERATION_INVALID",
        "OPENAPI_OPERATION_ID_INVALID",
        "OPENAPI_OPERATION_ID_DUPLICATE",
        "OPENAPI_RESPONSES_INVALID",
        "OPENAPI_COMPONENTS_INVALID",
    }

    empty_paths = copy.deepcopy(malformed)
    empty_paths["paths"] = []
    _, empty_issues = validate_openapi_shape(
        _replace_document(catalog, catalog.openapi_path, empty_paths)
    )
    assert "OPENAPI_PATHS_INVALID" in _issue_codes(empty_issues)


def test_fixture_matrix_reports_malformed_case_shapes(catalog: ContractCatalog) -> None:
    source = strict_json_load(MATRIX_PATH)
    base = copy.deepcopy(source["cases"][0])
    duplicate = copy.deepcopy(base)
    missing_rule = copy.deepcopy(base)
    missing_rule["id"] = "DATA-SCHEMA-999"
    missing_rule.pop("expected_rule")
    invalid_path = {
        "id": "DATA-SCHEMA-998",
        "schema": "",
        "instance": "",
        "valid": True,
        "expected_rule": "closed path",
    }
    matrix = {
        "format": "wrong-format",
        "cases": [None, {"id": "bad", "valid": "yes"}, base, duplicate, missing_rule, invalid_path],
        "semantic_cases": [],
    }

    result = validate_fixture_cases(catalog, matrix, MATRIX_PATH)
    codes = _issue_codes(result.issues)

    assert result.declared_cases == 6
    assert {
        "FIXTURE_MATRIX_FORMAT_INVALID",
        "FIXTURE_CASE_INVALID",
        "FIXTURE_CASE_ID_INVALID",
        "FIXTURE_EXPECTATION_INVALID",
        "FIXTURE_CASE_ID_DUPLICATE",
        "REFERENCE_PATH_INVALID",
        "FIXTURE_RULE_MISSING",
    } <= codes


def test_semantic_fixture_matrix_reports_binding_and_oracle_failures(
    catalog: ContractCatalog,
) -> None:
    source = strict_json_load(MATRIX_PATH)
    base = copy.deepcopy(source["semantic_cases"][0])
    duplicate = copy.deepcopy(base)
    mismatch = copy.deepcopy(base)
    mismatch["id"] = "DATA-SEM-994"
    mismatch["oracle"] = {}
    materialized = copy.deepcopy(
        next(case for case in source["semantic_cases"] if "materialized-manifest" in case["fixture"])
    )
    materialized["id"] = "DATA-SEM-993"
    materialized["oracle"] = {
        "manifest_sha256": "0" * 64,
        "files": -1,
        "clean_audit": {},
        "leaky_audit": {},
    }
    matrix = {
        "format": "llm-foundations-schema-validation-cases-v1",
        "cases": [],
        "semantic_cases": [
            None,
            {"id": "DATA-SEM-998", "status": "RUN", "fixture": 1, "oracle": []},
            base,
            duplicate,
            mismatch,
            {"id": "DATA-SEM-997", "status": "NOT_RUN", "fixture": "canonical-fixtures.json#missing", "oracle": {}},
            {"id": "DATA-SEM-996", "status": "NOT_RUN", "fixture": "validation-cases.json", "oracle": {}},
            {"id": "DATA-SEM-995", "status": "NOT_RUN", "fixture": "../outside.json", "oracle": {}},
            materialized,
        ],
    }

    result = validate_fixture_cases(catalog, matrix, MATRIX_PATH)
    codes = _issue_codes(result.issues)

    assert result.semantic_cases == 9
    assert {
        "SEMANTIC_CASE_INVALID",
        "SEMANTIC_CASE_STATUS_INVALID",
        "SEMANTIC_CASE_BINDING_INVALID",
        "SEMANTIC_CASE_ID_DUPLICATE",
        "SEMANTIC_FIXTURE_MISSING",
        "SEMANTIC_FIXTURE_BINDING_UNSUPPORTED",
        "REFERENCE_ESCAPE",
        "SEMANTIC_ORACLE_MISMATCH",
        "SEMANTIC_MANIFEST_DIGEST_MISMATCH",
        "SEMANTIC_MANIFEST_COUNT_MISMATCH",
        "SEMANTIC_AUDIT_MISMATCH",
    } <= codes


def test_model_profile_fixture_rejects_nonobject_and_identity_mismatch_manifests(
    catalog: ContractCatalog,
    tmp_path: Path,
) -> None:
    profile = strict_json_load(PROFILE_PATH)

    list_bytes = b"[]\n"
    list_manifest = tmp_path / "list-manifest.json"
    list_manifest.write_bytes(list_bytes)
    list_profile = copy.deepcopy(profile)
    list_profile["download_manifest_sha256"] = hashlib.sha256(list_bytes).hexdigest()
    list_issues = validate_model_profile_fixture(
        catalog, list_profile, PROFILE_PATH, list_manifest
    )
    assert "MODEL_DOWNLOAD_MANIFEST_TYPE_INVALID" in _issue_codes(list_issues)

    mismatched_manifest = {
        "model_profile_id": "different-profile",
        "repository": "different/repository",
        "revision": "different-revision",
        "license": "different-license",
    }
    raw = json.dumps(mismatched_manifest, sort_keys=True).encode("utf-8")
    identity_manifest = tmp_path / "identity-manifest.json"
    identity_manifest.write_bytes(raw)
    identity_profile = copy.deepcopy(profile)
    identity_profile["download_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    identity_issues = validate_model_profile_fixture(
        catalog, identity_profile, PROFILE_PATH, identity_manifest
    )
    assert "MODEL_PROFILE_MANIFEST_IDENTITY_MISMATCH" in _issue_codes(identity_issues)


def test_model_profile_validator_exceptions_fail_closed(
    catalog: ContractCatalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_validator(*_args: Any, **_kwargs: Any) -> tuple[str, ...]:
        raise RuntimeError("synthetic validator failure")

    monkeypatch.setattr(catalog, "schema_errors", fail_validator)
    issues = validate_model_profile_fixture(
        catalog,
        strict_json_load(PROFILE_PATH),
        PROFILE_PATH,
        MANIFEST_PATH,
    )
    assert _issue_codes(issues) == {"MODEL_PROFILE_VALIDATOR_ERROR"}


def test_error_vocabulary_rejects_every_ambiguous_binding_shape() -> None:
    def flat(code: str, kind: str, status: int | None, parent: str | None, retryable: Any) -> dict[str, Any]:
        return {
            "code": code,
            "kinds": [kind],
            "http_status": status,
            "top_level_code": parent,
            "retryable": retryable,
        }

    vocabulary = {
        "format": "wrong-format",
        "codes": [
            7,
            {"code": 7, "kinds": ["top_level"]},
            flat("DUPLICATE", "top_level", 400, None, False),
            flat("DUPLICATE", "top_level", 400, None, False),
            {"code": "NO_KINDS", "kinds": []},
            flat("lowercase", "top_level", 400, None, False),
            flat("UPPER_TERMINAL", "terminal", None, None, False),
            {"code": "MISSING_BINDING", "kinds": ["top_level"]},
            {"code": "AMBIGUOUS", "kinds": ["top_level", "reason"], "bindings": {}},
            {
                "code": "BAD_SHAPE",
                "kinds": ["top_level", "reason"],
                "bindings": {
                    "top_level": {},
                    "reason": {"http_status": 400, "top_level_code": "PARENT", "retryable": False},
                },
            },
            flat("BAD_RETRYABLE", "top_level", 400, None, "false"),
            flat("BAD_STATUS", "top_level", 200, None, False),
            flat("BAD_PARENT", "reason", 400, "lowercase", False),
            flat("BAD_JOB", "job", 400, None, False),
        ],
    }

    normalized, issues = error_bindings(vocabulary)
    codes = _issue_codes(issues)

    assert {"BAD_RETRYABLE", "BAD_STATUS", "BAD_PARENT", "BAD_JOB"} <= {
        code for code, _kind in normalized
    }
    assert {
        "ERROR_VOCABULARY_FORMAT_INVALID",
        "ERROR_CODE_ROW_INVALID",
        "ERROR_CODE_INVALID",
        "ERROR_CODE_DUPLICATE",
        "ERROR_CODE_KINDS_INVALID",
        "ERROR_BINDING_SHAPE_INVALID",
        "ERROR_BINDING_AMBIGUOUS",
        "ERROR_BINDING_RETRYABLE_INVALID",
        "ERROR_BINDING_ROLE_INVALID",
    } <= codes

    _, rows_issues = error_bindings(
        {"format": "llm-foundations-error-vocabulary-v2", "codes": {}}
    )
    assert _issue_codes(rows_issues) == {"ERROR_VOCABULARY_ROWS_INVALID"}


def test_semantic_binding_validation_rejects_rule_and_annotation_shapes(
    catalog: ContractCatalog,
) -> None:
    semantic_path = (catalog.contract_root / "semantic-rules.json").resolve()
    documents = dict(catalog.documents)
    documents[semantic_path] = {
        "rules": [
            None,
            {"id": "RULE", "applies_to": "component", "predicate": "true"},
            {"id": "RULE", "applies_to": 1},
        ]
    }
    openapi = copy.deepcopy(catalog.openapi)
    openapi["components"]["schemas"]["TinyTrainRequest"]["x-semantic-rules"] = [
        "RULE",
        "RULE",
    ]
    openapi["components"]["schemas"].pop("ReasonCode")
    documents[catalog.openapi_path] = openapi

    rule_count, _, _, issues = validate_semantic_and_error_bindings(
        replace(catalog, documents=documents)
    )
    codes = _issue_codes(issues)

    assert rule_count == 3
    assert {
        "SEMANTIC_RULE_INVALID",
        "SEMANTIC_RULE_DUPLICATE",
        "SEMANTIC_RULE_SHAPE_INVALID",
        "SEMANTIC_ANNOTATION_INVALID",
        "OPENAPI_ERROR_ENUM_MISSING",
    } <= codes

    documents[semantic_path] = {"rules": "not-an-array"}
    _, _, _, invalid_rules = validate_semantic_and_error_bindings(
        replace(catalog, documents=documents)
    )
    assert _issue_codes(invalid_rules) == {"SEMANTIC_RULES_INVALID"}


def test_semantic_binding_validation_detects_missing_parent_and_rejection_contradictions(
    catalog: ContractCatalog,
) -> None:
    semantic_path = (catalog.contract_root / "semantic-rules.json").resolve()
    vocabulary_path = (catalog.contract_root / "error-vocabulary.json").resolve()
    documents = dict(catalog.documents)
    documents[semantic_path] = {
        "rules": [
            {
                "id": "ONLY_RULE",
                "applies_to": "component",
                "predicate": "true",
                "rejection": {
                    "http_status": 400,
                    "code": "OTHER_TOP",
                    "reason_code": "ONLY_REASON",
                },
            }
        ]
    }
    documents[vocabulary_path] = {
        "format": "llm-foundations-error-vocabulary-v2",
        "codes": [
            {
                "code": "ONLY_REASON",
                "kinds": ["reason"],
                "http_status": 400,
                "top_level_code": "MISSING_TOP",
                "retryable": False,
            }
        ],
    }

    _, _, _, issues = validate_semantic_and_error_bindings(
        replace(catalog, documents=documents)
    )
    assert {
        "OPENAPI_ERROR_ENUM_MISMATCH",
        "ERROR_REASON_PARENT_MISSING",
        "SEMANTIC_REJECTION_TOP_LEVEL_CONTRADICTION",
        "SEMANTIC_REJECTION_REASON_CONTRADICTION",
    } <= _issue_codes(issues)


def test_runtime_json_entry_points_reject_non_json_python_values() -> None:
    with pytest.raises(TypeError):
        strict_json(1)  # type: ignore[arg-type]

    with pytest.raises(ApiError) as non_string_key:
        validate_schema({}, {1: "value"})  # type: ignore[dict-item]
    assert non_string_key.value.reason_code == "SCHEMA_INVALID"

    with pytest.raises(ApiError) as non_json_value:
        validate_schema({}, {"value": object()})  # type: ignore[dict-item]
    assert non_json_value.value.reason_code == "SCHEMA_INVALID"


def test_inline_schema_keywords_report_precise_paths_and_normalize_numbers() -> None:
    schema = {
        "type": "object",
        "minProperties": 6,
        "maxProperties": 6,
        "required": ["score", "text", "items", "mode", "dependent", "peer"],
        "properties": {
            "score": {
                "type": "number",
                "minimum": 0,
                "maximum": 8,
                "exclusiveMinimum": -1,
                "exclusiveMaximum": 9,
                "multipleOf": 2,
            },
            "text": {"type": "string", "minLength": 2, "maxLength": 3, "pattern": "^[a-z]+$"},
            "items": {
                "type": "array",
                "minItems": 2,
                "maxItems": 3,
                "uniqueItems": True,
                "contains": {"const": 5},
                "minContains": 1,
                "maxContains": 1,
                "prefixItems": [{"type": "integer"}],
                "items": {"type": "number"},
            },
            "mode": {"const": "x"},
            "dependent": {"type": "string"},
            "peer": {"type": "string"},
        },
        "additionalProperties": False,
        "propertyNames": {"type": "string", "pattern": "^[a-z]+$"},
        "dependentRequired": {"dependent": ["peer"]},
        "allOf": [
            {
                "if": {"properties": {"mode": {"const": "x"}}, "required": ["mode"]},
                "then": {"properties": {"peer": {"const": "ok"}}},
                "else": {"properties": {"peer": {"const": "other"}}},
            }
        ],
    }
    valid = {
        "score": 4,
        "text": "ok",
        "items": [1, 5],
        "mode": "x",
        "dependent": "present",
        "peer": "ok",
    }

    normalized = validate_schema(schema, valid)
    assert normalized["score"] == 4.0
    assert type(normalized["score"]) is float
    assert normalized["items"] == [1, 5.0]

    invalid = {
        "score": 10,
        "text": "TOOLONG",
        "items": [1, 1, 1, 1],
        "mode": "x",
        "dependent": "present",
        "BAD": True,
    }
    with pytest.raises(ApiError) as caught:
        validate_schema(schema, invalid)
    paths = {row["field_path"] for row in caught.value.field_errors}
    assert {
        "/BAD",
        "/items",
        "/items/1",
        "/peer",
        "/score",
        "/text",
    } <= paths


def test_inline_schema_composition_and_additional_values_are_normalized() -> None:
    schema = {
        "type": "object",
        "properties": {
            "kind": {"enum": ["a", "b"]},
            "head": {"type": "number"},
        },
        "additionalProperties": {"type": "number"},
        "allOf": [{"properties": {"head": {"type": "number"}}}],
        "oneOf": [
            {"properties": {"kind": {"const": "a"}}},
            {"properties": {"kind": {"const": "b"}}},
        ],
        "if": {"properties": {"kind": {"const": "a"}}},
        "then": {"properties": {"head": {"type": "number"}}},
    }
    normalized = validate_schema(schema, {"kind": "a", "head": 1, "tail": 2})
    assert normalized == {"kind": "a", "head": 1.0, "tail": 2.0}

    assert validate_schema({"type": ["null", "integer"]}, None) is None
    assert validate_schema(True, {"accepted": True}) == {"accepted": True}
    with pytest.raises(ApiError):
        validate_schema(False, None)
    with pytest.raises(ApiError):
        validate_schema({"oneOf": [{"type": "integer"}, {"type": "number"}]}, 1)
    with pytest.raises(ApiError):
        validate_schema({"anyOf": [{"type": "string"}, {"type": "integer"}]}, [])
    with pytest.raises(ApiError):
        validate_schema({"not": {"const": "blocked"}}, "blocked")


@pytest.mark.parametrize(
    ("schema", "value"),
    [
        ([True], None),
        ({"x-reason-code": 1}, None),
        ({"type": "made-up"}, None),
        ({"type": "string", "format": "email"}, "x"),
        ({"type": "string", "pattern": "["}, "x"),
        ({"type": "number", "multipleOf": 0}, 1),
        ({"type": "object", "required": "x"}, {}),
        ({"type": "object", "properties": []}, {}),
        ({"type": "object", "additionalProperties": 1}, {"extra": 1}),
        ({"type": "object", "dependentRequired": []}, {}),
        ({"type": "object", "dependentRequired": {"a": "b"}}, {"a": 1}),
        ({"type": "array", "prefixItems": {}}, []),
        ({"anyOf": []}, None),
        ({"oneOf": []}, None),
        ({"enum": []}, None),
        ({"madeUpKeyword": True}, None),
    ],
)
def test_malformed_inline_schema_definitions_fail_closed(schema: Any, value: Any) -> None:
    with pytest.raises(SchemaDefinitionError):
        validate_schema(schema, value)


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": 1},
        {"$ref": "a#b#c"},
        {"$ref": "#named-anchor"},
        {"$ref": "#/components/schemas/NoSuchComponent"},
        {"$ref": "#/paths/~2invalid"},
        {"$ref": "#/servers/99"},
        {"$ref": "#/info/title"},
    ],
)
def test_malformed_or_non_schema_references_fail_closed(schema: dict[str, Any]) -> None:
    with pytest.raises(SchemaDefinitionError):
        validate_schema(schema, None)


def test_runtime_schema_names_and_formats_are_closed() -> None:
    with pytest.raises(ValueError):
        validate("", {})
    with pytest.raises(ContractDataError):
        validate("does-not-exist.schema.json", {})

    uuid_schema = {"type": "string", "format": "uuid"}
    value = "123e4567-e89b-42d3-a456-426614174001"
    assert validate_schema(uuid_schema, value) == value
    for invalid in (value.upper(), "not-a-uuid"):
        with pytest.raises(ApiError):
            validate_schema(uuid_schema, invalid)
    with pytest.raises(ApiError):
        validate_schema({"type": "string", "format": "date-time"}, "2026-02-30T00:00:00.000Z")
    assert validate_schema({"type": "string", "format": "binary"}, "opaque") == "opaque"


def test_inline_schema_reports_minimum_shape_and_multiple_boundaries() -> None:
    schema = {
        "type": "object",
        "minProperties": 3,
        "properties": {
            "score": {"type": "number", "multipleOf": 2},
            "text": {"type": "string", "minLength": 2},
        },
    }
    with pytest.raises(ApiError) as caught:
        validate_schema(schema, {"score": 3, "text": "x"})
    assert {item["field_path"] for item in caught.value.field_errors} == {
        "",
        "/score",
        "/text",
    }


def test_semantic_rules_accept_structurally_inapplicable_and_exact_valid_inputs() -> None:
    assert validate_schema(
        {"x-semantic-rules": ["TINY_EVAL_CADENCE"]}, []
    ) == []
    assert validate_schema(
        {"type": "object", "x-semantic-rules": ["CHAT_MESSAGE_BYTES"]},
        {"backend": "chat"},
    ) == {"backend": "chat"}
    subjects = [
        {"kind": "base_model"},
        {"kind": "adapter"},
    ]
    assert validate_schema(
        {"type": "object", "x-semantic-rules": ["EVALUATE_SUBJECT_PROFILE"]},
        {
            "evaluation_profile_id": "applied-intents-greedy-v1",
            "subjects": subjects,
            "release_token": "present",
        },
    )["subjects"] == subjects
    with pytest.raises(ApiError) as empty_selection:
        validate_schema(
            {"type": "object", "x-semantic-rules": ["BACKUP_EXPORT_SELECTION"]},
            {"note_ids": []},
        )
    assert empty_selection.value.reason_code == "SEMANTIC_INVALID"


def test_materialized_fixture_manifest_reports_custody_failures(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "materialized-manifest.json"
    invalid_utf8 = tmp_path / "invalid-utf8.jsonl"
    invalid_utf8.write_bytes(b"\xff")
    with pytest.raises(ContractSourceError) as unreadable:
        contracts_module._load_jsonl_strict(invalid_utf8)
    assert unreadable.value.code == "FIXTURE_SOURCE_UNREADABLE"

    no_newline = tmp_path / "no-newline.jsonl"
    no_newline.write_bytes(b"{}")
    with pytest.raises(ContractSourceError) as newline:
        contracts_module._load_jsonl_strict(no_newline)
    assert newline.value.code == "FIXTURE_JSONL_NEWLINE"

    assert _issue_codes(
        contracts_module._validate_materialized_manifest(
            manifest_path, {}, tmp_path
        )[1]
    ) == {"FIXTURE_MANIFEST_FORMAT_INVALID"}
    assert _issue_codes(
        contracts_module._validate_materialized_manifest(
            manifest_path,
            {"format": "llm-foundations-materialized-fixtures-v1", "files": {}},
            tmp_path,
        )[1]
    ) == {"FIXTURE_MANIFEST_FILES_INVALID"}

    valid = tmp_path / "valid.jsonl"
    valid.write_bytes(b'{"record":1}\n')
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_bytes(b'{bad}\n')
    valid_row = {
        "path": valid.name,
        "sha256": "0" * 64,
        "utf8_bytes": 0,
        "records": 0,
    }
    malformed_raw = malformed.read_bytes()
    manifest = {
        "format": "llm-foundations-materialized-fixtures-v1",
        "files": [
            None,
            valid_row,
            dict(valid_row),
            {"path": "../escape.jsonl"},
            {"path": "missing.jsonl"},
            {
                "path": malformed.name,
                "sha256": hashlib.sha256(malformed_raw).hexdigest(),
                "utf8_bytes": len(malformed_raw),
                "records": 1,
            },
        ],
    }
    count, issues = contracts_module._validate_materialized_manifest(
        manifest_path, manifest, tmp_path
    )
    assert count == 6
    assert {
        "FIXTURE_MANIFEST_ROW_INVALID",
        "FIXTURE_MANIFEST_PATH_DUPLICATE",
        "REFERENCE_ESCAPE",
        "FIXTURE_MATERIALIZED_MISSING",
        "FIXTURE_MATERIALIZED_IDENTITY_MISMATCH",
        "JSON_INVALID",
        "FIXTURE_RECORD_COUNT_MISMATCH",
    } <= _issue_codes(issues)


SEMANTIC_FAILURES = [
    ("DIAGNOSIS_HOLD_SHAPE", {"exercise_profile_id": "wrong"}, "/curriculum_hold_after_step", "DIAGNOSIS_HOLD_INVALID"),
    ("CHAT_MESSAGE_BYTES", {"messages": [{"content": "x" * (LIMITS["chat_message_bytes"] + 1)}]}, "/messages/0/content", "PAYLOAD_TOO_LARGE"),
    ("CHAT_TOTAL_BYTES", {"messages": [{"content": "x" * (LIMITS["chat_total_bytes"] + 1)}]}, "/messages", "PAYLOAD_TOO_LARGE"),
    ("TINY_PROMPT_BYTES", {"prompt": "x" * (LIMITS["tiny_prompt_bytes"] + 1)}, "/prompt", "PAYLOAD_TOO_LARGE"),
    ("RETRIEVAL_QUERY_BYTES", {"query": "x" * (LIMITS["retrieval_query_bytes"] + 1)}, "/query", "PAYLOAD_TOO_LARGE"),
    ("EXPORT_SELECTION_COUNT", {}, "/checkpoint_options", "SEMANTIC_INVALID"),
    ("EXPORT_RESUME_REQUIRES_DATASET", {"checkpoint_options": [{"include_resume_state": True}], "content_options": {}}, "/content_options/include_dataset_bytes", "SEMANTIC_INVALID"),
    ("EVALUATE_SUBJECT_PROFILE", {"evaluation_profile_id": "wrong", "subjects": []}, "/subjects", "SUBJECT_INCOMPATIBLE"),
    ("PROGRESS_DIRECT_FLAGS", {"dimension": "practiced", "value": True}, "/dimension", "PROGRESS_DERIVED_FLAG"),
    ("BACKUP_EXPORT_SELECTION", {"include_notes": True, "confirm_sensitive_text": False}, "/confirm_sensitive_text", "SEMANTIC_INVALID"),
    ("JOB_FAILED_REASON", {"state": "failed", "terminal_reason": "A", "error": {"code": "B"}}, "/terminal_reason", "SEMANTIC_INVALID"),
    ("JOB_STEP_BOUNDS", {"step": 3, "requested_final_step": 2}, "/step", "SEMANTIC_INVALID"),
    ("JOB_TIME_ORDER", {"created_at": "2026-10-01T02:00:00.000Z", "updated_at": "2026-10-01T01:00:00.000Z"}, "/updated_at", "SEMANTIC_INVALID"),
    ("ASSESSMENT_TOTAL", {"rubric_rows": [{"score": 2}, {"score": 3}], "total_score": 4}, "/total_score", "SEMANTIC_INVALID"),
    ("PROGRESS_TIMESTAMPS", {"modules": {"m1": {"read": True}}}, "/modules/m1/read_at", "SEMANTIC_INVALID"),
    ("EVIDENCE_IMPORT_STATE", {"claim_status": "imported_claim", "verification_status": "artifact_verified"}, "/claim_status", "SEMANTIC_INVALID"),
    ("EVIDENCE_VERIFIED_STATE", {"verification_status": "artifact_verified", "claim_status": "imported_claim", "verification_checks": []}, "/verification_status", "SEMANTIC_INVALID"),
    ("TINY_SAVE_CADENCE", {"backend": "tiny_v2", "training_identity": {"save_every": 2, "eval_every": 1}}, "/training_identity/save_every", "SEMANTIC_INVALID"),
    ("BUNDLE_UNIQUE_PATHS", {"entries": [{"path": "A.txt"}, {"path": "a.TXT"}]}, "/entries/1/path", "ARCHIVE_UNSAFE"),
    ("BUNDLE_NO_ABSOLUTE_PATHS", {"entries": [{"path": "C:/secret"}]}, "/entries", "SEMANTIC_INVALID"),
]


@pytest.mark.parametrize(("rule", "value", "path", "reason"), SEMANTIC_FAILURES)
def test_pure_semantic_rules_reject_structured_edge_cases(
    rule: str,
    value: dict[str, Any],
    path: str,
    reason: str,
) -> None:
    schema = {"type": "object", "x-semantic-rules": [rule]}
    with pytest.raises(ApiError) as caught:
        validate_schema(schema, value)
    assert caught.value.reason_code == reason
    assert caught.value.field_errors[0]["field_path"] == path

    assert validate_schema(schema, value, include_semantic=False) == value


def test_semantic_rule_declarations_are_closed_but_deferred_rules_remain_admissible() -> None:
    with pytest.raises(SchemaDefinitionError, match="x-semantic-rules"):
        validate_schema({"x-semantic-rules": "TINY_EVAL_CADENCE"}, {})
    with pytest.raises(SchemaDefinitionError, match="unsupported semantic rule"):
        validate_schema({"x-semantic-rules": ["NO_SUCH_RULE"]}, {})

    value = {"arbitrary": "data"}
    assert validate_schema(
        {"type": "object", "x-semantic-rules": ["TINY_PARAMETER_CAP"]},
        value,
    ) == value
