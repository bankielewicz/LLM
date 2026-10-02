"""Validate the approved APP-009 authority overlay without rewriting the sealed spec."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any


SIDECAR_REL = "docs/implementation/s2/authority.json"
AMENDMENT_REL = "docs/specs/intermediate-v1-amendments/APP-009-revision-1.3.md"
PROPOSAL_REL = "docs/implementation/s2/APP-009-amendment-proposal.md"
S0_AUTHORITY_REL = "docs/implementation/s0/authority.json"
S2_BASELINE_REL = "docs/implementation/s2/baseline.json"
SPEC_ROOT_REL = "docs/specs/intermediate-v1"
SPEC_MANIFEST_REL = SPEC_ROOT_REL + "/spec-manifest.json"
TARGET_REL = SPEC_ROOT_REL + "/06-APPLIED-MODELS.md"
OPENAPI_REL = SPEC_ROOT_REL + "/contracts/openapi.json"
ACCEPTANCE_REL = SPEC_ROOT_REL + "/acceptance-cases.json"

EXPECTED_SIDECAR_SHA256 = "b57f1f81a3d94bd3a6ef7194da1b9ac7b58222f49eb84742839a319e6b5978f4"
EXPECTED_AMENDMENT_SHA256 = "4559e05d354cb938dd4e65955b66ff2b3ab6a9073b538d1e98e536d8ff5eaad9"
EXPECTED_PROPOSAL_SHA256 = "88e4458774e3c0fa38e92042d1c604fc2804b09dc8b396bcdf509ab425189a7c"
EXPECTED_S0_AUTHORITY_SHA256 = "c79bb229e3563fa2b6c0e641653218cdd3444f10bb0b6ed8e74ef56561bf4357"
EXPECTED_S2_BASELINE_SHA256 = "7b607dd7405d139bba30b81e757c4fa97f5461d64f32599557bab8a239fe9914"
EXPECTED_SPEC_MANIFEST_SHA256 = "7526190e853580adcc3681c9ac0559be4ae8c741878b5a686bdcf624d9f131b8"
EXPECTED_SPEC_PAYLOAD_SHA256 = "974306b73634d9d073fb90de941fe55c8fb83cd073a585613cb6249b06691571"
EXPECTED_TARGET_SHA256 = "a7143c0ef83302f7569f0931b9b86b52380cf9a47ebf0f2a5011a924a4be5a13"
EXPECTED_OPENAPI_SHA256 = "e0602505eb4f9381695950c2ed1610ec74e0a0166014de6006e1dc05e2053585"
EXPECTED_ACCEPTANCE_SHA256 = "fe8caec52e1be042f61ce10af81046ad3bff833af33c31526f226893713cf7e4"

BASE_REVISION = "1.2"
EFFECTIVE_REVISION = "1.3"
BASE_FILE_COUNT = 150
ACCEPTANCE_CASE_COUNT = 128
ACCEPTANCE_EXECUTION_UNIT_COUNT = 605
_PRODUCT_STATUS = "NOT_RUN"
_TICK = chr(96)
OLD_PHRASE = (
    f'Its request uses {_TICK}backend: "tiny"{_TICK} with the prospective '
    f'{_TICK}model_id{_TICK}, prompt, and decoding fields'
)
NEW_PHRASE = (
    f'Its request uses {_TICK}backend: "tiny"{_TICK} with the prospective '
    f'{_TICK}checkpoint_id{_TICK}, prompt, and decoding fields'
)


class AuthorityError(ValueError):
    """The APP-009 authority overlay or its immutable base is invalid."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise AuthorityError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise AuthorityError(f"non-finite JSON constant: {value}")


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuthorityError(f"{label} is not UTF-8") from exc
    try:
        value = json.loads(
            decoded,
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, AuthorityError) as exc:
        raise AuthorityError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AuthorityError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuthorityError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise AuthorityError(
            f"{label} fields differ: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )
    return value


def _expect(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise AuthorityError(f"{label} differs: expected {expected!r}, found {actual!r}")


def _contained(root: Path, relative: str, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise AuthorityError(f"{label} path must be a nonempty string")
    if "\\" in relative:
        raise AuthorityError(f"{label} path must use POSIX separators")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or relative != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise AuthorityError(f"{label} path is not normalized and relative: {relative!r}")

    root = root.resolve(strict=True)
    candidate = root.joinpath(*pure.parts)
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise AuthorityError(f"{label} path crosses a symlink: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise AuthorityError(f"{label} path is missing: {relative}") from exc
    if not resolved.is_relative_to(root):
        raise AuthorityError(f"{label} path escapes the repository: {relative}")
    if not resolved.is_file():
        raise AuthorityError(f"{label} path is not a regular file: {relative}")
    return resolved


def _read_bound(
    repo: Path,
    relative: str,
    expected_sha256: str,
    label: str,
) -> tuple[Path, bytes]:
    path = _contained(repo, relative, label)
    raw = path.read_bytes()
    actual = _sha256(raw)
    if actual != expected_sha256:
        raise AuthorityError(
            f"{label} SHA-256 differs: expected {expected_sha256}, found {actual}"
        )
    return path, raw


def _validate_sidecar(value: dict[str, Any]) -> None:
    _exact_keys(
        value,
        {
            "amendment",
            "authorization",
            "base_authority",
            "effective_revision",
            "format",
            "preserved_contracts",
        },
        "sidecar",
    )
    _expect(value["format"], "llm-foundations-s2-authority-amendment-v1", "sidecar format")
    _expect(value["effective_revision"], EFFECTIVE_REVISION, "effective revision")

    amendment = _exact_keys(
        value["amendment"], {"id", "path", "sha256", "target"}, "amendment"
    )
    _expect(amendment["id"], "APP-009-revision-1.3", "amendment id")
    _expect(amendment["path"], AMENDMENT_REL, "amendment path")
    _expect(amendment["sha256"], EXPECTED_AMENDMENT_SHA256, "amendment SHA-256")
    target = _exact_keys(
        amendment["target"],
        {
            "base_path",
            "base_sha256",
            "new_phrase",
            "new_phrase_occurrences_in_base",
            "old_phrase",
            "old_phrase_occurrences_in_base",
        },
        "amendment target",
    )
    _expect(target["base_path"], TARGET_REL, "amendment target path")
    _expect(target["base_sha256"], EXPECTED_TARGET_SHA256, "amendment target SHA-256")
    _expect(target["old_phrase"], OLD_PHRASE, "old phrase")
    _expect(target["new_phrase"], NEW_PHRASE, "new phrase")
    _expect(target["old_phrase_occurrences_in_base"], 1, "old phrase occurrence claim")
    _expect(target["new_phrase_occurrences_in_base"], 0, "new phrase occurrence claim")

    authorization = _exact_keys(
        value["authorization"],
        {"approved_on", "proposal_path", "proposal_sha256", "status"},
        "authorization",
    )
    _expect(authorization["approved_on"], "2026-10-01", "authorization date")
    _expect(authorization["proposal_path"], PROPOSAL_REL, "proposal path")
    _expect(authorization["proposal_sha256"], EXPECTED_PROPOSAL_SHA256, "proposal SHA-256")
    _expect(authorization["status"], "APPROVED", "authorization status")

    base = _exact_keys(
        value["base_authority"],
        {
            "file_count",
            "manifest_path",
            "manifest_sha256",
            "payload_sha256",
            "revision",
            "s0_authority_path",
            "s0_authority_sha256",
            "s2_baseline_path",
            "s2_baseline_sha256",
        },
        "base authority",
    )
    expected_base = {
        "file_count": BASE_FILE_COUNT,
        "manifest_path": SPEC_MANIFEST_REL,
        "manifest_sha256": EXPECTED_SPEC_MANIFEST_SHA256,
        "payload_sha256": EXPECTED_SPEC_PAYLOAD_SHA256,
        "revision": BASE_REVISION,
        "s0_authority_path": S0_AUTHORITY_REL,
        "s0_authority_sha256": EXPECTED_S0_AUTHORITY_SHA256,
        "s2_baseline_path": S2_BASELINE_REL,
        "s2_baseline_sha256": EXPECTED_S2_BASELINE_SHA256,
    }
    for key, expected in expected_base.items():
        _expect(base[key], expected, f"base authority {key}")

    preserved = _exact_keys(
        value["preserved_contracts"],
        {
            "acceptance_execution_unit_count",
            "acceptance_registry_path",
            "acceptance_registry_sha256",
            "openapi_path",
            "openapi_sha256",
            "product_qualification",
        },
        "preserved contracts",
    )
    expected_preserved = {
        "acceptance_execution_unit_count": ACCEPTANCE_EXECUTION_UNIT_COUNT,
        "acceptance_registry_path": ACCEPTANCE_REL,
        "acceptance_registry_sha256": EXPECTED_ACCEPTANCE_SHA256,
        "openapi_path": OPENAPI_REL,
        "openapi_sha256": EXPECTED_OPENAPI_SHA256,
        "product_qualification": _PRODUCT_STATUS,
    }
    for key, expected in expected_preserved.items():
        _expect(preserved[key], expected, f"preserved contracts {key}")


def _validate_manifest(repo: Path, value: dict[str, Any]) -> None:
    _exact_keys(
        value,
        {
            "base_commit",
            "digest_rule",
            "exclusions",
            "files",
            "format",
            "payload_sha256",
            "spec_revision",
        },
        "base specification manifest",
    )
    _expect(value["format"], "llm-foundations-spec-manifest-v1", "manifest format")
    _expect(value["spec_revision"], BASE_REVISION, "manifest revision")
    _expect(value["payload_sha256"], EXPECTED_SPEC_PAYLOAD_SHA256, "manifest payload")
    files = value["files"]
    if not isinstance(files, list) or len(files) != BASE_FILE_COUNT:
        raise AuthorityError("base specification manifest does not contain exactly 150 files")
    paths: list[str] = []
    spec_root = _contained(repo, SPEC_ROOT_REL + "/README.md", "specification root marker").parent
    for index, row_value in enumerate(files):
        row = _exact_keys(
            row_value, {"path", "sha256", "size_bytes"}, f"manifest file {index}"
        )
        relative = row["path"]
        if not isinstance(relative, str):
            raise AuthorityError(f"manifest file {index} path is not a string")
        paths.append(relative)
        path = _contained(spec_root, relative, f"manifest file {relative}")
        raw = path.read_bytes()
        if len(raw) != row["size_bytes"] or _sha256(raw) != row["sha256"]:
            raise AuthorityError(f"base manifest member differs: {relative}")
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise AuthorityError("base specification manifest paths are not sorted and unique")
    _expect(_sha256(_canonical(files)), EXPECTED_SPEC_PAYLOAD_SHA256, "recomputed payload")


def _validate_s0_authority(value: dict[str, Any]) -> None:
    active = value.get("active_specification")
    if not isinstance(active, dict):
        raise AuthorityError("S0 authority lacks active_specification")
    expected = {
        "revision": BASE_REVISION,
        "spec_root": SPEC_ROOT_REL,
        "file_count": BASE_FILE_COUNT,
        "manifest_sha256": EXPECTED_SPEC_MANIFEST_SHA256,
        "payload_sha256": EXPECTED_SPEC_PAYLOAD_SHA256,
    }
    for key, item in expected.items():
        _expect(active.get(key), item, f"S0 active specification {key}")
    _expect(
        value.get("acceptance_execution_unit_count"),
        ACCEPTANCE_EXECUTION_UNIT_COUNT,
        "S0 acceptance denominator",
    )
    _expect(value.get("product_qualification"), _PRODUCT_STATUS, "S0 product status")


def _validate_s2_baseline(value: dict[str, Any]) -> None:
    expected = {
        "format": "llm-foundations-s2-baseline-v1",
        "authority_revision": BASE_REVISION,
        "authority_payload_sha256": EXPECTED_SPEC_PAYLOAD_SHA256,
        "sealed_specification_files": BASE_FILE_COUNT,
        "product_acceptance_execution_units": ACCEPTANCE_EXECUTION_UNIT_COUNT,
        "product_qualification": _PRODUCT_STATUS,
    }
    for key, item in expected.items():
        _expect(value.get(key), item, f"S2 baseline {key}")
    rows = value.get("files")
    if not isinstance(rows, list):
        raise AuthorityError("S2 baseline files must be an array")
    bound = {
        row.get("path"): row.get("sha256")
        for row in rows
        if isinstance(row, dict)
    }
    _expect(bound.get(S0_AUTHORITY_REL), EXPECTED_S0_AUTHORITY_SHA256, "baseline S0 authority")
    _expect(bound.get(SPEC_MANIFEST_REL), EXPECTED_SPEC_MANIFEST_SHA256, "baseline spec manifest")
    _expect(bound.get(ACCEPTANCE_REL), EXPECTED_ACCEPTANCE_SHA256, "baseline acceptance registry")


def _validate_openapi(value: dict[str, Any]) -> None:
    try:
        tiny = value["components"]["schemas"]["ContextPreviewRequest"]["oneOf"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise AuthorityError("OpenAPI lacks the tiny ContextPreviewRequest branch") from exc
    properties = tiny.get("properties")
    required = tiny.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise AuthorityError("tiny ContextPreviewRequest branch is malformed")
    if (
        "checkpoint_id" not in properties
        or "checkpoint_id" not in required
        or "model_id" in properties
        or "model_id" in required
    ):
        raise AuthorityError(
            "tiny ContextPreviewRequest is not checkpoint_id-only at the public request boundary"
        )


def _validate_acceptance(value: dict[str, Any]) -> None:
    _exact_keys(
        value,
        {"cases", "execution_units", "product_status", "spec_revision"},
        "acceptance registry",
    )
    _expect(value["spec_revision"], BASE_REVISION, "acceptance registry revision")
    _expect(value["product_status"], _PRODUCT_STATUS, "acceptance product status")
    cases = value["cases"]
    units = value["execution_units"]
    if not isinstance(cases, list) or len(cases) != ACCEPTANCE_CASE_COUNT:
        raise AuthorityError("acceptance case denominator differs from 128")
    if not isinstance(units, list) or len(units) != ACCEPTANCE_EXECUTION_UNIT_COUNT:
        raise AuthorityError("acceptance execution-unit denominator differs from 605")
    if any(not isinstance(row, dict) or row.get("status") != _PRODUCT_STATUS for row in cases):
        raise AuthorityError("an acceptance case was promoted from NOT_RUN")
    if any(not isinstance(row, dict) or row.get("status") != _PRODUCT_STATUS for row in units):
        raise AuthorityError("an acceptance execution unit was promoted from NOT_RUN")


def audit(repo: Path) -> dict[str, object]:
    """Return the exact effective authority only when every binding validates."""

    repo = Path(repo).resolve(strict=True)
    if not repo.is_dir():
        raise AuthorityError("repository root is not a directory")

    sidecar_path = _contained(repo, SIDECAR_REL, "authority sidecar")
    sidecar_raw = sidecar_path.read_bytes()
    sidecar = _strict_json(sidecar_raw, "authority sidecar")
    _validate_sidecar(sidecar)

    _, manifest_raw = _read_bound(
        repo, SPEC_MANIFEST_REL, EXPECTED_SPEC_MANIFEST_SHA256, "base specification manifest"
    )
    _validate_manifest(repo, _strict_json(manifest_raw, "base specification manifest"))

    _, s0_raw = _read_bound(
        repo, S0_AUTHORITY_REL, EXPECTED_S0_AUTHORITY_SHA256, "S0 authority"
    )
    _validate_s0_authority(_strict_json(s0_raw, "S0 authority"))

    _, baseline_raw = _read_bound(
        repo, S2_BASELINE_REL, EXPECTED_S2_BASELINE_SHA256, "S2 baseline"
    )
    _validate_s2_baseline(_strict_json(baseline_raw, "S2 baseline"))

    _read_bound(repo, PROPOSAL_REL, EXPECTED_PROPOSAL_SHA256, "approved proposal")
    _, amendment_raw = _read_bound(
        repo, AMENDMENT_REL, EXPECTED_AMENDMENT_SHA256, "normative amendment"
    )
    amendment_text = amendment_raw.decode("utf-8")
    if amendment_text.count(OLD_PHRASE) != 1 or amendment_text.count(NEW_PHRASE) != 1:
        raise AuthorityError("normative amendment does not contain one exact old/new phrase pair")

    _, target_raw = _read_bound(
        repo, TARGET_REL, EXPECTED_TARGET_SHA256, "sealed APP-009 target"
    )
    target_text = target_raw.decode("utf-8")
    if target_text.count(OLD_PHRASE) != 1 or target_text.count(NEW_PHRASE) != 0:
        raise AuthorityError("sealed APP-009 target does not have the exact 1-to-0 phrase boundary")

    _, openapi_raw = _read_bound(
        repo, OPENAPI_REL, EXPECTED_OPENAPI_SHA256, "preserved OpenAPI"
    )
    _validate_openapi(_strict_json(openapi_raw, "preserved OpenAPI"))

    _, acceptance_raw = _read_bound(
        repo, ACCEPTANCE_REL, EXPECTED_ACCEPTANCE_SHA256, "acceptance registry"
    )
    _validate_acceptance(_strict_json(acceptance_raw, "acceptance registry"))

    actual_sidecar_sha256 = _sha256(sidecar_raw)
    _expect(actual_sidecar_sha256, EXPECTED_SIDECAR_SHA256, "authority sidecar SHA-256")

    return {
        "format": "llm-foundations-s2-authority-audit-v1",
        "status": "PASS",
        "effective_revision": EFFECTIVE_REVISION,
        "base": {
            "revision": BASE_REVISION,
            "file_count": BASE_FILE_COUNT,
            "manifest_sha256": EXPECTED_SPEC_MANIFEST_SHA256,
            "payload_sha256": EXPECTED_SPEC_PAYLOAD_SHA256,
        },
        "amendment": {
            "id": "APP-009-revision-1.3",
            "path": AMENDMENT_REL,
            "sha256": EXPECTED_AMENDMENT_SHA256,
            "proposal_sha256": EXPECTED_PROPOSAL_SHA256,
            "old_phrase_occurrences_in_base": 1,
            "new_phrase_occurrences_in_base": 0,
        },
        "authority_sidecar_sha256": actual_sidecar_sha256,
        "acceptance": {
            "cases": ACCEPTANCE_CASE_COUNT,
            "execution_units": ACCEPTANCE_EXECUTION_UNIT_COUNT,
            "status": _PRODUCT_STATUS,
        },
        "public_tiny_preview_subject": "checkpoint_id",
        "product_qualification": _PRODUCT_STATUS,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root; defaults to the checkout containing this script.",
    )
    args = parser.parse_args(argv)
    try:
        result = audit(args.repo)
    except (AuthorityError, OSError) as exc:
        print(f"S2 authority FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
