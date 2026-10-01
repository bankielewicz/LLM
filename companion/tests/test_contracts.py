from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from llm_foundations_companion.contracts import (
    ContractCatalog,
    ContractSourceError,
    error_bindings,
    main,
    strict_json_load,
    strict_json_loads,
    validate_contract_package,
    validate_fixture_cases,
    validate_model_profile_fixture,
    validate_reference_closure,
    validate_semantic_and_error_bindings,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SPEC_ROOT = REPOSITORY_ROOT / "docs" / "specs" / "intermediate-v1"


@pytest.fixture(scope="module")
def catalog() -> ContractCatalog:
    return ContractCatalog.from_spec_root(SPEC_ROOT)


def test_authoritative_contract_package_passes_closed_validation() -> None:
    report = validate_contract_package(SPEC_ROOT)

    assert report.ok, [issue.as_dict() for issue in report.issues]
    assert report.contract_json_documents >= 34
    assert report.schema_units >= 137
    assert report.references >= 445
    assert report.operations == 57
    assert report.fixture_cases == report.fixture_cases_matched == 49
    assert report.model_profile_fixtures == 1
    assert report.model_profile_fixtures_matched == 1
    assert report.semantic_fixture_cases == 3
    assert report.semantic_fixture_bindings >= 8
    assert report.materialized_fixture_files == 14
    assert report.semantic_rules == 33
    assert report.semantic_rule_bindings == 25
    assert report.error_codes == 104
    assert (
        "fixtures/applied/smollm2-135m-instruct-v1.json"
        in report.source_sha256
    )
    assert (
        "fixtures/applied/model-download-manifest.json"
        in report.source_sha256
    )


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ('{"key": 1, "key": 2}', "JSON_DUPLICATE_KEY"),
        ('{"value": NaN}', "JSON_NON_FINITE"),
        ('{"value": Infinity}', "JSON_NON_FINITE"),
        ('{"value": -Infinity}', "JSON_NON_FINITE"),
        ('{"value": 1e9999}', "JSON_NON_FINITE"),
    ],
)
def test_strict_json_rejects_ambiguous_or_nonfinite_values(
    payload: str,
    code: str,
) -> None:
    with pytest.raises(ContractSourceError) as caught:
        strict_json_loads(payload)

    assert caught.value.code == code


def test_reference_closure_rejects_network_and_missing_targets(
    tmp_path: Path,
) -> None:
    contract_root = tmp_path / "contracts"
    contract_root.mkdir()
    owner = contract_root / "owner.json"
    document = {
        "$defs": {"local": {"type": "string"}},
        "local": {"$ref": "#/$defs/local"},
        "missing": {"$ref": "missing.schema.json"},
        "network": {"$ref": "https://example.invalid/schema.json"},
    }
    count, issues = validate_reference_closure(
        contract_root,
        {owner: document},
    )

    assert count == 3
    assert {issue.code for issue in issues} == {
        "REFERENCE_NETWORK_FORBIDDEN",
        "REFERENCE_UNRESOLVED",
    }


def test_reference_closure_rejects_noncanonical_array_indices_and_bad_escape(
    tmp_path: Path,
) -> None:
    contract_root = tmp_path / "contracts"
    contract_root.mkdir()
    owner = contract_root / "owner.json"
    document = {
        "items": ["zero"],
        "leading_zero": {"$ref": "#/items/00"},
        "non_ascii": {"$ref": "#/items/٠"},
        "out_of_bounds": {"$ref": "#/items/1"},
        "invalid_escape": {"$ref": "#/items/~2"},
    }

    count, issues = validate_reference_closure(
        contract_root,
        {owner: document},
    )

    assert count == 4
    assert sum(issue.code == "REFERENCE_POINTER_INVALID" for issue in issues) == 3
    assert sum(issue.code == "REFERENCE_UNRESOLVED" for issue in issues) == 1


def test_registered_schema_accepts_valid_and_rejects_invalid_fixture(
    catalog: ContractCatalog,
) -> None:
    schema = (
        SPEC_ROOT
        / "contracts"
        / "schemas"
        / "artifact-descriptor.schema.json"
    )
    fixture_root = SPEC_ROOT / "fixtures" / "data"
    valid = strict_json_load(fixture_root / "valid-artifact.json")
    invalid = strict_json_load(fixture_root / "invalid-artifact-path.json")

    assert catalog.schema_errors(schema, valid) == ()
    assert catalog.schema_errors(schema, invalid)


def test_fixture_matrix_detects_contradictory_expectation(
    catalog: ContractCatalog,
) -> None:
    matrix_path = SPEC_ROOT / "fixtures" / "data" / "validation-cases.json"
    matrix = strict_json_load(matrix_path)
    contradictory = copy.deepcopy(matrix)
    contradictory["cases"][0]["valid"] = not contradictory["cases"][0]["valid"]

    result = validate_fixture_cases(catalog, contradictory, matrix_path)

    mismatches = [
        issue
        for issue in result.issues
        if issue.code == "FIXTURE_EXPECTATION_MISMATCH"
    ]
    assert len(mismatches) == 1
    assert "DATA-SCHEMA-001" in mismatches[0].message


def test_model_profile_rejects_malformed_instance_and_stale_manifest_digest(
    catalog: ContractCatalog,
    tmp_path: Path,
) -> None:
    profile_path = (
        SPEC_ROOT / "fixtures" / "applied" / "smollm2-135m-instruct-v1.json"
    )
    manifest_path = (
        SPEC_ROOT / "fixtures" / "applied" / "model-download-manifest.json"
    )
    profile = strict_json_load(profile_path)

    malformed = copy.deepcopy(profile)
    malformed.pop("architecture")
    malformed_issues = validate_model_profile_fixture(
        catalog,
        malformed,
        profile_path,
        manifest_path,
    )
    assert any(
        issue.code == "MODEL_PROFILE_SCHEMA_REJECTED"
        for issue in malformed_issues
    )

    stale_manifest = tmp_path / "model-download-manifest.json"
    stale_manifest.write_bytes(manifest_path.read_bytes() + b" ")
    stale_issues = validate_model_profile_fixture(
        catalog,
        profile,
        profile_path,
        stale_manifest,
    )
    assert any(
        issue.code == "MODEL_PROFILE_MANIFEST_DIGEST_MISMATCH"
        for issue in stale_issues
    )


def test_error_vocabulary_has_distinct_payload_boundaries(
    catalog: ContractCatalog,
) -> None:
    vocabulary_path = SPEC_ROOT / "contracts" / "error-vocabulary.json"
    vocabulary = strict_json_load(vocabulary_path)
    bindings, issues = error_bindings(vocabulary)

    assert issues == ()
    assert bindings[("PAYLOAD_TOO_LARGE", "top_level")] == {
        "http_status": 413,
        "top_level_code": None,
        "retryable": False,
    }
    assert bindings[("PAYLOAD_TOO_LARGE", "reason")] == {
        "http_status": 400,
        "top_level_code": "VALIDATION_FAILED",
        "retryable": False,
    }

    contradictory = copy.deepcopy(vocabulary)
    payload = next(
        row for row in contradictory["codes"] if row["code"] == "PAYLOAD_TOO_LARGE"
    )
    payload["bindings"]["reason"]["http_status"] = 413
    documents = dict(catalog.documents)
    documents[vocabulary_path.resolve()] = contradictory
    mutated_catalog = replace(catalog, documents=documents)

    _, _, _, detected = validate_semantic_and_error_bindings(mutated_catalog)
    codes = {issue.code for issue in detected}
    assert "ERROR_REASON_PARENT_CONTRADICTION" in codes
    assert "SEMANTIC_REJECTION_REASON_CONTRADICTION" in codes


def test_unknown_semantic_rule_annotation_is_rejected(
    catalog: ContractCatalog,
) -> None:
    openapi = copy.deepcopy(catalog.openapi)
    openapi["components"]["schemas"]["TinyTrainRequest"][
        "x-semantic-rules"
    ].append("UNKNOWN_RULE")
    documents = dict(catalog.documents)
    documents[catalog.openapi_path] = openapi
    mutated_catalog = replace(catalog, documents=documents)

    _, _, _, issues = validate_semantic_and_error_bindings(mutated_catalog)

    assert any(issue.code == "SEMANTIC_RULE_UNRESOLVED" for issue in issues)


def test_cli_emits_machine_readable_counts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["--spec-root", str(SPEC_ROOT)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "PASS"
    assert payload["counts"]["fixture_cases"] == 49
    assert payload["counts"]["fixture_cases_matched"] == 49
    assert payload["counts"]["model_profile_fixtures_matched"] == 1
    assert payload["counts"]["materialized_fixture_files"] == 14
