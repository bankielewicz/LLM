"""Offline validation for the frozen intermediate-v1 contract package.

This module is an S0 development and build-time boundary.  It reads the
authoritative specification in place; it does not start the companion, copy
contract files into a wheel, or make a product-qualification claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlsplit

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012


_HTTP_METHODS = {
    "delete",
    "get",
    "head",
    "options",
    "patch",
    "post",
    "put",
    "trace",
}
_JSON_SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_TERMINAL_CODE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SEMANTIC_CASE_RE = re.compile(r"^DATA-SEM-[0-9]{3}$")
_SCHEMA_CASE_RE = re.compile(r"^DATA-SCHEMA-[0-9]{3}$")
_POINTER_ESCAPE_RE = re.compile(r"~(?![01])")
_ARRAY_INDEX_RE = re.compile(r"^(?:0|[1-9][0-9]*)$", re.ASCII)
_FORMAT_CHECKER = FormatChecker()


class ContractSourceError(ValueError):
    """A deterministic failure to read or resolve authoritative contract data."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        source: Path | None = None,
        location: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.source = source
        self.location = location


@dataclass(frozen=True)
class ValidationIssue:
    """One stable contract validation diagnostic."""

    code: str
    source: str
    location: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "source": self.source,
            "location": self.location,
            "message": self.message,
        }


@dataclass(frozen=True)
class FixtureValidation:
    declared_cases: int
    matched_cases: int
    semantic_cases: int
    semantic_binding_checks: int
    materialized_files: int
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True)
class ContractValidationReport:
    spec_root: str
    contract_json_documents: int
    schema_units: int
    references: int
    operations: int
    fixture_cases: int
    fixture_cases_matched: int
    model_profile_fixtures: int
    model_profile_fixtures_matched: int
    semantic_fixture_cases: int
    semantic_fixture_bindings: int
    materialized_fixture_files: int
    semantic_rules: int
    semantic_rule_bindings: int
    error_codes: int
    source_sha256: Mapping[str, str]
    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "PASS" if self.ok else "FAIL",
            "spec_root": self.spec_root,
            "counts": {
                "contract_json_documents": self.contract_json_documents,
                "schema_units": self.schema_units,
                "references": self.references,
                "operations": self.operations,
                "fixture_cases": self.fixture_cases,
                "fixture_cases_matched": self.fixture_cases_matched,
                "model_profile_fixtures": self.model_profile_fixtures,
                "model_profile_fixtures_matched": (
                    self.model_profile_fixtures_matched
                ),
                "semantic_fixture_cases": self.semantic_fixture_cases,
                "semantic_fixture_bindings": self.semantic_fixture_bindings,
                "materialized_fixture_files": self.materialized_fixture_files,
                "semantic_rules": self.semantic_rules,
                "semantic_rule_bindings": self.semantic_rule_bindings,
                "error_codes": self.error_codes,
            },
            "source_sha256": dict(sorted(self.source_sha256.items())),
            "issues": [issue.as_dict() for issue in self.issues],
            "limitations": [
                "S0 contract validation only; no service, browser, model, or training execution.",
                "Fixture PASS means declared schema/oracle relationships match, not product acceptance.",
            ],
        }


class _DuplicateKey(ValueError):
    pass


class _NonFiniteNumber(ValueError):
    pass


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _strict_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise _NonFiniteNumber(token)
    return value


def _reject_constant(token: str) -> Any:
    raise _NonFiniteNumber(token)


def strict_json_loads(text: str, *, source: Path | None = None) -> Any:
    """Parse JSON while rejecting duplicate keys and every non-finite number."""

    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
            parse_float=_strict_float,
        )
    except _DuplicateKey as exc:
        raise ContractSourceError(
            "JSON_DUPLICATE_KEY",
            f"duplicate JSON object key: {exc}",
            source=source,
        ) from exc
    except _NonFiniteNumber as exc:
        raise ContractSourceError(
            "JSON_NON_FINITE",
            f"non-finite JSON number: {exc}",
            source=source,
        ) from exc
    except json.JSONDecodeError as exc:
        raise ContractSourceError(
            "JSON_INVALID",
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            source=source,
        ) from exc


def strict_json_load(path: Path) -> Any:
    """Read strict UTF-8 JSON from *path* without BOM or replacement decoding."""

    path = path.resolve()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ContractSourceError(
            "SOURCE_UNREADABLE",
            f"cannot read contract source: {exc.strerror or type(exc).__name__}",
            source=path,
        ) from exc
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ContractSourceError(
            "JSON_ENCODING_INVALID",
            f"contract source is not strict UTF-8 at byte {exc.start}",
            source=path,
        ) from exc
    return strict_json_loads(text, source=path)


def _source_name(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _issue(
    code: str,
    source: Path | str,
    location: str,
    message: str,
    *,
    root: Path | None = None,
) -> ValidationIssue:
    if isinstance(source, Path):
        source_text = _source_name(source, root) if root else source.as_posix()
    else:
        source_text = source
    return ValidationIssue(code, source_text, location, message)


def _walk_key(value: Any, key: str, pointer: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            escaped = child_key.replace("~", "~0").replace("/", "~1")
            child_pointer = f"{pointer}/{escaped}"
            if child_key == key:
                yield child_pointer, child
            yield from _walk_key(child, key, child_pointer)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_key(child, key, f"{pointer}/{index}")


def _pointer_get(document: Any, fragment: str, *, source: Path) -> Any:
    if fragment in ("", "#"):
        return document
    decoded = unquote(fragment[1:] if fragment.startswith("#") else fragment)
    if not decoded.startswith("/"):
        raise ContractSourceError(
            "REFERENCE_FRAGMENT_UNSUPPORTED",
            "reference fragment must be an RFC 6901 JSON Pointer",
            source=source,
            location=fragment,
        )
    current = document
    for raw_part in decoded[1:].split("/"):
        if _POINTER_ESCAPE_RE.search(raw_part):
            raise ContractSourceError(
                "REFERENCE_POINTER_INVALID",
                f"invalid JSON Pointer escape in {raw_part!r}",
                source=source,
                location=fragment,
            )
        part = raw_part.replace("~1", "/").replace("~0", "~")
        try:
            if isinstance(current, list):
                if not _ARRAY_INDEX_RE.fullmatch(part):
                    raise ContractSourceError(
                        "REFERENCE_POINTER_INVALID",
                        (
                            "array index must be canonical ASCII 0 or "
                            f"[1-9][0-9]*: {part!r}"
                        ),
                        source=source,
                        location=fragment,
                    )
                current = current[int(part)]
            else:
                current = current[part]
        except (IndexError, KeyError, TypeError) as exc:
            raise ContractSourceError(
                "REFERENCE_UNRESOLVED",
                f"JSON Pointer does not resolve: {fragment}",
                source=source,
                location=fragment,
            ) from exc
    return current


def _bounded_path(base: Path, raw: str, allowed_root: Path, *, source: Path) -> Path:
    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc or parsed.query:
        raise ContractSourceError(
            "REFERENCE_NETWORK_FORBIDDEN",
            f"only relative, query-free local references are allowed: {raw}",
            source=source,
            location=raw,
        )
    path_text = unquote(parsed.path)
    windows_path = PureWindowsPath(path_text)
    if (
        not path_text
        or path_text.startswith(("/", "\\"))
        or windows_path.is_absolute()
        or windows_path.drive
    ):
        raise ContractSourceError(
            "REFERENCE_PATH_INVALID",
            f"reference path must be relative: {raw}",
            source=source,
            location=raw,
        )
    target = (base / path_text).resolve()
    try:
        target.relative_to(allowed_root.resolve())
    except ValueError as exc:
        raise ContractSourceError(
            "REFERENCE_ESCAPE",
            f"reference escapes the allowed root: {raw}",
            source=source,
            location=raw,
        ) from exc
    return target


def resolve_local_reference(
    owner_path: Path,
    owner_document: Any,
    reference: str,
    contract_root: Path,
    documents: Mapping[Path, Any] | None = None,
) -> Any:
    """Resolve one closed local reference and reject network or escaping targets."""

    if not isinstance(reference, str):
        raise ContractSourceError(
            "REFERENCE_TYPE_INVALID",
            "$ref must be a string",
            source=owner_path,
        )
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc or parsed.query:
        raise ContractSourceError(
            "REFERENCE_NETWORK_FORBIDDEN",
            f"network, absolute URI, and query references are forbidden: {reference}",
            source=owner_path,
            location=reference,
        )
    if parsed.path:
        target_path = _bounded_path(
            owner_path.parent,
            parsed.path,
            contract_root,
            source=owner_path,
        )
        if not target_path.is_file():
            raise ContractSourceError(
                "REFERENCE_UNRESOLVED",
                f"referenced file does not exist: {reference}",
                source=owner_path,
                location=reference,
            )
        target_document = (
            documents.get(target_path)
            if documents is not None and target_path in documents
            else strict_json_load(target_path)
        )
    else:
        target_path = owner_path
        target_document = owner_document
    return _pointer_get(
        target_document,
        f"#{parsed.fragment}" if parsed.fragment else "",
        source=target_path,
    )


@dataclass
class ContractCatalog:
    """Strictly loaded local contract documents and their offline registry."""

    spec_root: Path
    contract_root: Path
    documents: dict[Path, Any]
    schema_paths: tuple[Path, ...]
    openapi_path: Path
    model_profile_path: Path
    registry: Registry

    @property
    def openapi(self) -> dict[str, Any]:
        value = self.documents[self.openapi_path]
        if not isinstance(value, dict):
            raise ContractSourceError(
                "OPENAPI_TYPE_INVALID",
                "openapi.json must contain an object",
                source=self.openapi_path,
            )
        return value

    @classmethod
    def from_spec_root(cls, spec_root: Path) -> "ContractCatalog":
        spec_root = spec_root.resolve()
        contract_root = (spec_root / "contracts").resolve()
        if not contract_root.is_dir():
            raise ContractSourceError(
                "CONTRACT_ROOT_MISSING",
                "spec root has no contracts directory",
                source=contract_root,
            )
        paths = tuple(sorted(contract_root.rglob("*.json")))
        documents = {path.resolve(): strict_json_load(path) for path in paths}
        openapi_path = (contract_root / "openapi.json").resolve()
        model_profile_path = (contract_root / "model-profile.json").resolve()
        for required in (openapi_path, model_profile_path):
            if required not in documents:
                raise ContractSourceError(
                    "CONTRACT_SOURCE_MISSING",
                    f"required contract file is missing: {required.name}",
                    source=required,
                )
        schema_paths = tuple(
            path.resolve()
            for path in sorted((contract_root / "schemas").glob("*.schema.json"))
        )
        registry = Registry()
        for path in (openapi_path, model_profile_path, *schema_paths):
            document = documents[path]
            registry = registry.with_resource(
                path.as_uri(),
                Resource(contents=document, specification=DRAFT202012),
            )
        return cls(
            spec_root=spec_root,
            contract_root=contract_root,
            documents=documents,
            schema_paths=schema_paths,
            openapi_path=openapi_path,
            model_profile_path=model_profile_path,
            registry=registry,
        )

    def schema_errors(self, schema_path: Path, instance: Any) -> tuple[str, ...]:
        path = schema_path.resolve()
        if path not in self.schema_paths and path != self.model_profile_path:
            raise ContractSourceError(
                "SCHEMA_NOT_REGISTERED",
                f"schema is outside the registered catalog: {path.name}",
                source=path,
            )
        wrapper = {"$schema": _JSON_SCHEMA_URI, "$ref": path.as_uri()}
        validator = Draft202012Validator(
            wrapper,
            registry=self.registry,
            format_checker=_FORMAT_CHECKER,
        )
        return _render_schema_errors(validator.iter_errors(instance))

    def component_errors(self, component: str, instance: Any) -> tuple[str, ...]:
        schemas = self.openapi.get("components", {}).get("schemas", {})
        if component not in schemas:
            raise ContractSourceError(
                "OPENAPI_COMPONENT_MISSING",
                f"OpenAPI schema component does not exist: {component}",
                source=self.openapi_path,
            )
        uri = f"{self.openapi_path.as_uri()}#/components/schemas/{component}"
        validator = Draft202012Validator(
            {"$schema": _JSON_SCHEMA_URI, "$ref": uri},
            registry=self.registry,
            format_checker=_FORMAT_CHECKER,
        )
        return _render_schema_errors(validator.iter_errors(instance))


def _render_schema_errors(errors: Iterable[Any]) -> tuple[str, ...]:
    ordered = sorted(
        errors,
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            error.message,
        ),
    )
    return tuple(
        f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: "
        f"{error.message}"
        for error in ordered
    )


def validate_reference_closure(
    contract_root: Path,
    documents: Mapping[Path, Any],
) -> tuple[int, tuple[ValidationIssue, ...]]:
    """Resolve every $ref from its owning document without a retrieval callback."""

    root = contract_root.resolve()
    normalized = {path.resolve(): value for path, value in documents.items()}
    issues: list[ValidationIssue] = []
    count = 0
    for owner_path, document in sorted(
        normalized.items(), key=lambda item: item[0].as_posix()
    ):
        for location, reference in _walk_key(document, "$ref"):
            count += 1
            try:
                resolve_local_reference(
                    owner_path,
                    document,
                    reference,
                    root,
                    normalized,
                )
            except ContractSourceError as exc:
                issues.append(
                    _issue(
                        exc.code,
                        owner_path,
                        location,
                        str(exc),
                        root=root,
                    )
                )
    return count, tuple(issues)


def _check_schema(
    schema: Any,
    *,
    source: Path,
    location: str,
    spec_root: Path,
) -> ValidationIssue | None:
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        return _issue(
            "SCHEMA_INVALID",
            source,
            location,
            str(exc),
            root=spec_root,
        )
    return None


def validate_schema_units(
    catalog: ContractCatalog,
) -> tuple[int, tuple[ValidationIssue, ...]]:
    issues: list[ValidationIssue] = []
    count = 0
    for path in (*catalog.schema_paths, catalog.model_profile_path):
        count += 1
        issue = _check_schema(
            catalog.documents[path],
            source=path,
            location="",
            spec_root=catalog.spec_root,
        )
        if issue:
            issues.append(issue)
    schemas = catalog.openapi.get("components", {}).get("schemas", {})
    if isinstance(schemas, dict):
        for name, schema in sorted(schemas.items()):
            count += 1
            issue = _check_schema(
                schema,
                source=catalog.openapi_path,
                location=f"/components/schemas/{name}",
                spec_root=catalog.spec_root,
            )
            if issue:
                issues.append(issue)
    for location, schema in _walk_key(catalog.openapi.get("paths", {}), "schema"):
        count += 1
        issue = _check_schema(
            schema,
            source=catalog.openapi_path,
            location=f"/paths{location}",
            spec_root=catalog.spec_root,
        )
        if issue:
            issues.append(issue)
    return count, tuple(issues)


def validate_openapi_shape(
    catalog: ContractCatalog,
) -> tuple[int, tuple[ValidationIssue, ...]]:
    document = catalog.openapi
    issues: list[ValidationIssue] = []
    source = catalog.openapi_path
    if document.get("openapi") != "3.1.0":
        issues.append(
            _issue(
                "OPENAPI_VERSION_INVALID",
                source,
                "/openapi",
                "the frozen contract requires OpenAPI 3.1.0",
                root=catalog.spec_root,
            )
        )
    info = document.get("info")
    if not isinstance(info, dict) or not all(
        isinstance(info.get(key), str) and info[key] for key in ("title", "version")
    ):
        issues.append(
            _issue(
                "OPENAPI_INFO_INVALID",
                source,
                "/info",
                "info.title and info.version must be non-empty strings",
                root=catalog.spec_root,
            )
        )
    servers = document.get("servers")
    if servers != [{"url": "http://127.0.0.1:8765"}]:
        issues.append(
            _issue(
                "OPENAPI_SERVER_INVALID",
                source,
                "/servers",
                "the contract must expose only the fixed loopback server",
                root=catalog.spec_root,
            )
        )
    paths = document.get("paths")
    if not isinstance(paths, dict) or not paths:
        issues.append(
            _issue(
                "OPENAPI_PATHS_INVALID",
                source,
                "/paths",
                "paths must be a non-empty object",
                root=catalog.spec_root,
            )
        )
        return 0, tuple(issues)
    operation_ids: dict[str, str] = {}
    operation_count = 0
    for path_name, path_item in sorted(paths.items()):
        location = f"/paths/{path_name.replace('~', '~0').replace('/', '~1')}"
        if not isinstance(path_name, str) or not path_name.startswith("/"):
            issues.append(
                _issue(
                    "OPENAPI_PATH_INVALID",
                    source,
                    location,
                    "path keys must begin with /",
                    root=catalog.spec_root,
                )
            )
        if not isinstance(path_item, dict):
            issues.append(
                _issue(
                    "OPENAPI_PATH_ITEM_INVALID",
                    source,
                    location,
                    "path item must be an object",
                    root=catalog.spec_root,
                )
            )
            continue
        for method, operation in sorted(path_item.items()):
            if method not in _HTTP_METHODS:
                continue
            operation_count += 1
            op_location = f"{location}/{method}"
            if not isinstance(operation, dict):
                issues.append(
                    _issue(
                        "OPENAPI_OPERATION_INVALID",
                        source,
                        op_location,
                        "operation must be an object",
                        root=catalog.spec_root,
                    )
                )
                continue
            operation_id = operation.get("operationId")
            if not isinstance(operation_id, str) or not operation_id:
                issues.append(
                    _issue(
                        "OPENAPI_OPERATION_ID_INVALID",
                        source,
                        f"{op_location}/operationId",
                        "operationId must be a non-empty string",
                        root=catalog.spec_root,
                    )
                )
            elif operation_id in operation_ids:
                issues.append(
                    _issue(
                        "OPENAPI_OPERATION_ID_DUPLICATE",
                        source,
                        f"{op_location}/operationId",
                        f"operationId duplicates {operation_ids[operation_id]}",
                        root=catalog.spec_root,
                    )
                )
            else:
                operation_ids[operation_id] = op_location
            responses = operation.get("responses")
            if not isinstance(responses, dict) or not responses:
                issues.append(
                    _issue(
                        "OPENAPI_RESPONSES_INVALID",
                        source,
                        f"{op_location}/responses",
                        "operation must define at least one response",
                        root=catalog.spec_root,
                    )
                )
    schemas = document.get("components", {}).get("schemas")
    if not isinstance(schemas, dict) or not schemas:
        issues.append(
            _issue(
                "OPENAPI_COMPONENTS_INVALID",
                source,
                "/components/schemas",
                "components.schemas must be a non-empty object",
                root=catalog.spec_root,
            )
        )
    return operation_count, tuple(issues)


def _fixture_case_issue(
    code: str,
    matrix_path: Path,
    case_index: int,
    message: str,
    spec_root: Path,
) -> ValidationIssue:
    return _issue(
        code,
        matrix_path,
        f"/cases/{case_index}",
        message,
        root=spec_root,
    )


def _load_jsonl_strict(path: Path) -> list[Any]:
    try:
        text = path.read_bytes().decode("utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as exc:
        raise ContractSourceError(
            "FIXTURE_SOURCE_UNREADABLE",
            f"cannot read strict UTF-8 JSONL fixture: {type(exc).__name__}",
            source=path,
        ) from exc
    lines = text.splitlines()
    if text and not text.endswith("\n"):
        raise ContractSourceError(
            "FIXTURE_JSONL_NEWLINE",
            "materialized JSONL fixture must end with LF",
            source=path,
        )
    return [
        strict_json_loads(line, source=path)
        for line in lines
        if line
    ]


def _validate_materialized_manifest(
    manifest_path: Path,
    manifest: Any,
    spec_root: Path,
) -> tuple[int, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    if not isinstance(manifest, dict) or manifest.get("format") != (
        "llm-foundations-materialized-fixtures-v1"
    ):
        return 0, [
            _issue(
                "FIXTURE_MANIFEST_FORMAT_INVALID",
                manifest_path,
                "/format",
                "unexpected materialized fixture manifest format",
                root=spec_root,
            )
        ]
    rows = manifest.get("files")
    if not isinstance(rows, list):
        return 0, [
            _issue(
                "FIXTURE_MANIFEST_FILES_INVALID",
                manifest_path,
                "/files",
                "manifest files must be an array",
                root=spec_root,
            )
        ]
    seen: set[str] = set()
    for index, row in enumerate(rows):
        location = f"/files/{index}"
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            issues.append(
                _issue(
                    "FIXTURE_MANIFEST_ROW_INVALID",
                    manifest_path,
                    location,
                    "manifest row must contain a string path",
                    root=spec_root,
                )
            )
            continue
        relative = row["path"]
        if relative in seen:
            issues.append(
                _issue(
                    "FIXTURE_MANIFEST_PATH_DUPLICATE",
                    manifest_path,
                    f"{location}/path",
                    f"duplicate materialized fixture path: {relative}",
                    root=spec_root,
                )
            )
            continue
        seen.add(relative)
        try:
            fixture_path = _bounded_path(
                manifest_path.parent,
                relative,
                manifest_path.parent,
                source=manifest_path,
            )
        except ContractSourceError as exc:
            issues.append(
                _issue(
                    exc.code,
                    manifest_path,
                    f"{location}/path",
                    str(exc),
                    root=spec_root,
                )
            )
            continue
        if not fixture_path.is_file():
            issues.append(
                _issue(
                    "FIXTURE_MATERIALIZED_MISSING",
                    manifest_path,
                    f"{location}/path",
                    f"materialized fixture is missing: {relative}",
                    root=spec_root,
                )
            )
            continue
        raw = fixture_path.read_bytes()
        actual_sha = hashlib.sha256(raw).hexdigest()
        if row.get("sha256") != actual_sha or row.get("utf8_bytes") != len(raw):
            issues.append(
                _issue(
                    "FIXTURE_MATERIALIZED_IDENTITY_MISMATCH",
                    manifest_path,
                    location,
                    f"retained bytes or SHA-256 differ for {relative}",
                    root=spec_root,
                )
            )
        try:
            records = _load_jsonl_strict(fixture_path)
        except ContractSourceError as exc:
            issues.append(
                _issue(
                    exc.code,
                    fixture_path,
                    "",
                    str(exc),
                    root=spec_root,
                )
            )
            continue
        if row.get("records") != len(records):
            issues.append(
                _issue(
                    "FIXTURE_RECORD_COUNT_MISMATCH",
                    manifest_path,
                    f"{location}/records",
                    f"declared and actual record counts differ for {relative}",
                    root=spec_root,
                )
            )
    return len(rows), issues


def _validate_semantic_fixture_cases(
    matrix_path: Path,
    matrix: Mapping[str, Any],
    spec_root: Path,
) -> tuple[int, int, int, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    cases = matrix.get("semantic_cases")
    if not isinstance(cases, list):
        return 0, 0, 0, [
            _issue(
                "SEMANTIC_CASES_INVALID",
                matrix_path,
                "/semantic_cases",
                "semantic_cases must be an array",
                root=spec_root,
            )
        ]
    data_root = matrix_path.parent.resolve()
    canonical_path = (data_root / "canonical-fixtures.json").resolve()
    canonical = strict_json_load(canonical_path)
    fixtures = canonical.get("fixtures", []) if isinstance(canonical, dict) else []
    by_id = {
        row.get("fixture_id"): row
        for row in fixtures
        if isinstance(row, dict) and isinstance(row.get("fixture_id"), str)
    }
    seen: set[str] = set()
    binding_checks = 0
    materialized_files = 0
    for index, case in enumerate(cases):
        location = f"/semantic_cases/{index}"
        if not isinstance(case, dict):
            issues.append(
                _issue(
                    "SEMANTIC_CASE_INVALID",
                    matrix_path,
                    location,
                    "semantic case must be an object",
                    root=spec_root,
                )
            )
            continue
        case_id = case.get("id")
        if not isinstance(case_id, str) or not _SEMANTIC_CASE_RE.fullmatch(case_id):
            issues.append(
                _issue(
                    "SEMANTIC_CASE_ID_INVALID",
                    matrix_path,
                    f"{location}/id",
                    "semantic case ID must match DATA-SEM-NNN",
                    root=spec_root,
                )
            )
        elif case_id in seen:
            issues.append(
                _issue(
                    "SEMANTIC_CASE_ID_DUPLICATE",
                    matrix_path,
                    f"{location}/id",
                    f"duplicate semantic case ID: {case_id}",
                    root=spec_root,
                )
            )
        else:
            seen.add(case_id)
        if case.get("status") != "NOT_RUN":
            issues.append(
                _issue(
                    "SEMANTIC_CASE_STATUS_INVALID",
                    matrix_path,
                    f"{location}/status",
                    "authored semantic fixtures remain NOT_RUN product evidence",
                    root=spec_root,
                )
            )
        fixture_ref = case.get("fixture")
        oracle = case.get("oracle")
        if not isinstance(fixture_ref, str) or not isinstance(oracle, dict):
            issues.append(
                _issue(
                    "SEMANTIC_CASE_BINDING_INVALID",
                    matrix_path,
                    location,
                    "semantic case needs string fixture and object oracle",
                    root=spec_root,
                )
            )
            continue
        filename, marker, selector = fixture_ref.partition("#")
        try:
            source_path = _bounded_path(
                data_root,
                filename,
                data_root,
                source=matrix_path,
            )
        except ContractSourceError as exc:
            issues.append(
                _issue(
                    exc.code,
                    matrix_path,
                    f"{location}/fixture",
                    str(exc),
                    root=spec_root,
                )
            )
            continue
        if source_path.name == "canonical-fixtures.json" and marker:
            source_fixture = by_id.get(selector)
            if source_fixture is None:
                issues.append(
                    _issue(
                        "SEMANTIC_FIXTURE_MISSING",
                        matrix_path,
                        f"{location}/fixture",
                        f"canonical fixture does not exist: {selector}",
                        root=spec_root,
                    )
                )
                continue
            if isinstance(source_fixture.get("expected_audit"), dict):
                binding_checks += 1
                if oracle != source_fixture["expected_audit"]:
                    issues.append(
                        _issue(
                            "SEMANTIC_ORACLE_MISMATCH",
                            matrix_path,
                            f"{location}/oracle",
                            f"oracle differs from {selector}.expected_audit",
                            root=spec_root,
                        )
                    )
            elif isinstance(source_fixture.get("records"), dict):
                for key, expected in source_fixture["records"].items():
                    binding_checks += 1
                    if oracle.get(key) != expected:
                        issues.append(
                            _issue(
                                "SEMANTIC_ORACLE_MISMATCH",
                                matrix_path,
                                f"{location}/oracle/{key}",
                                f"oracle differs from {selector}.records.{key}",
                                root=spec_root,
                            )
                        )
        elif source_path.name == "materialized-manifest.json" and not marker:
            manifest = strict_json_load(source_path)
            materialized_files, manifest_issues = _validate_materialized_manifest(
                source_path,
                manifest,
                spec_root,
            )
            issues.extend(manifest_issues)
            binding_checks += 1
            actual_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if oracle.get("manifest_sha256") != actual_digest:
                issues.append(
                    _issue(
                        "SEMANTIC_MANIFEST_DIGEST_MISMATCH",
                        matrix_path,
                        f"{location}/oracle/manifest_sha256",
                        "semantic oracle does not bind the retained manifest bytes",
                        root=spec_root,
                    )
                )
            binding_checks += 1
            if oracle.get("files") != len(manifest.get("files", [])):
                issues.append(
                    _issue(
                        "SEMANTIC_MANIFEST_COUNT_MISMATCH",
                        matrix_path,
                        f"{location}/oracle/files",
                        "semantic oracle file count differs from the manifest",
                        root=spec_root,
                    )
                )
            for oracle_key, fixture_id in (
                ("clean_audit", "data-clinic-v1"),
                ("leaky_audit", "data-clinic-leaky-v1"),
            ):
                binding_checks += 1
                actual = manifest.get("audits", {}).get(fixture_id)
                if oracle.get(oracle_key) != actual:
                    issues.append(
                        _issue(
                            "SEMANTIC_AUDIT_MISMATCH",
                            matrix_path,
                            f"{location}/oracle/{oracle_key}",
                            f"semantic oracle differs from audits.{fixture_id}",
                            root=spec_root,
                        )
                    )
        else:
            issues.append(
                _issue(
                    "SEMANTIC_FIXTURE_BINDING_UNSUPPORTED",
                    matrix_path,
                    f"{location}/fixture",
                    f"no closed binding rule for fixture reference: {fixture_ref}",
                    root=spec_root,
                )
            )
    return len(cases), binding_checks, materialized_files, issues


def validate_fixture_cases(
    catalog: ContractCatalog,
    matrix: Mapping[str, Any],
    matrix_path: Path,
) -> FixtureValidation:
    """Validate schema fixtures plus the mechanically bound semantic oracles."""

    issues: list[ValidationIssue] = []
    if matrix.get("format") != "llm-foundations-schema-validation-cases-v1":
        issues.append(
            _issue(
                "FIXTURE_MATRIX_FORMAT_INVALID",
                matrix_path,
                "/format",
                "unexpected validation-case matrix format",
                root=catalog.spec_root,
            )
        )
    cases = matrix.get("cases")
    if not isinstance(cases, list):
        issues.append(
            _issue(
                "FIXTURE_CASES_INVALID",
                matrix_path,
                "/cases",
                "cases must be an array",
                root=catalog.spec_root,
            )
        )
        cases = []
    seen: set[str] = set()
    matched = 0
    schema_root = (catalog.contract_root / "schemas").resolve()
    data_root = matrix_path.parent.resolve()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            issues.append(
                _fixture_case_issue(
                    "FIXTURE_CASE_INVALID",
                    matrix_path,
                    index,
                    "fixture case must be an object",
                    catalog.spec_root,
                )
            )
            continue
        case_id = case.get("id")
        if not isinstance(case_id, str) or not _SCHEMA_CASE_RE.fullmatch(case_id):
            issues.append(
                _fixture_case_issue(
                    "FIXTURE_CASE_ID_INVALID",
                    matrix_path,
                    index,
                    "fixture ID must match DATA-SCHEMA-NNN",
                    catalog.spec_root,
                )
            )
        elif case_id in seen:
            issues.append(
                _fixture_case_issue(
                    "FIXTURE_CASE_ID_DUPLICATE",
                    matrix_path,
                    index,
                    f"duplicate fixture ID: {case_id}",
                    catalog.spec_root,
                )
            )
        else:
            seen.add(case_id)
        expected = case.get("valid")
        if not isinstance(expected, bool):
            issues.append(
                _fixture_case_issue(
                    "FIXTURE_EXPECTATION_INVALID",
                    matrix_path,
                    index,
                    "valid must be a boolean",
                    catalog.spec_root,
                )
            )
            continue
        try:
            schema_path = _bounded_path(
                data_root,
                case.get("schema", ""),
                schema_root,
                source=matrix_path,
            )
            instance_path = _bounded_path(
                data_root,
                case.get("instance", ""),
                data_root,
                source=matrix_path,
            )
            if schema_path not in catalog.schema_paths:
                raise ContractSourceError(
                    "SCHEMA_NOT_REGISTERED",
                    f"fixture schema is not registered: {schema_path.name}",
                    source=matrix_path,
                )
            instance = strict_json_load(instance_path)
            actual = not catalog.schema_errors(schema_path, instance)
        except ContractSourceError as exc:
            issues.append(
                _fixture_case_issue(
                    exc.code,
                    matrix_path,
                    index,
                    str(exc),
                    catalog.spec_root,
                )
            )
            continue
        except Exception as exc:
            issues.append(
                _fixture_case_issue(
                    "FIXTURE_VALIDATOR_ERROR",
                    matrix_path,
                    index,
                    f"validator failed closed: {type(exc).__name__}: {exc}",
                    catalog.spec_root,
                )
            )
            continue
        if actual != expected:
            issues.append(
                _fixture_case_issue(
                    "FIXTURE_EXPECTATION_MISMATCH",
                    matrix_path,
                    index,
                    f"{case_id} expected valid={expected}, observed valid={actual}",
                    catalog.spec_root,
                )
            )
        else:
            matched += 1
        if not isinstance(case.get("expected_rule"), str) or not case["expected_rule"]:
            issues.append(
                _fixture_case_issue(
                    "FIXTURE_RULE_MISSING",
                    matrix_path,
                    index,
                    "expected_rule must be a non-empty string",
                    catalog.spec_root,
                )
            )
    semantic_cases, binding_checks, materialized_files, semantic_issues = (
        _validate_semantic_fixture_cases(matrix_path, matrix, catalog.spec_root)
    )
    issues.extend(semantic_issues)
    return FixtureValidation(
        declared_cases=len(cases),
        matched_cases=matched,
        semantic_cases=semantic_cases,
        semantic_binding_checks=binding_checks,
        materialized_files=materialized_files,
        issues=tuple(issues),
    )


def validate_model_profile_fixture(
    catalog: ContractCatalog,
    profile: Any,
    profile_path: Path,
    manifest_path: Path,
) -> tuple[ValidationIssue, ...]:
    """Validate the pinned model profile and its exact download-manifest binding."""

    issues: list[ValidationIssue] = []
    try:
        schema_errors = catalog.schema_errors(catalog.model_profile_path, profile)
    except Exception as exc:
        issues.append(
            _issue(
                "MODEL_PROFILE_VALIDATOR_ERROR",
                profile_path,
                "",
                f"model-profile validator failed closed: {type(exc).__name__}: {exc}",
                root=catalog.spec_root,
            )
        )
        return tuple(issues)
    if schema_errors:
        issues.append(
            _issue(
                "MODEL_PROFILE_SCHEMA_REJECTED",
                profile_path,
                "",
                schema_errors[0],
                root=catalog.spec_root,
            )
        )

    manifest = strict_json_load(manifest_path)
    if not isinstance(profile, dict):
        return tuple(issues)
    actual_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if profile.get("download_manifest_sha256") != actual_digest:
        issues.append(
            _issue(
                "MODEL_PROFILE_MANIFEST_DIGEST_MISMATCH",
                profile_path,
                "/download_manifest_sha256",
                "profile digest does not bind the exact model-download-manifest bytes",
                root=catalog.spec_root,
            )
        )
    if isinstance(manifest, dict):
        identity_fields = ("model_profile_id", "repository", "revision", "license")
        mismatched = [
            field
            for field in identity_fields
            if profile.get(field) != manifest.get(field)
        ]
        if mismatched:
            issues.append(
                _issue(
                    "MODEL_PROFILE_MANIFEST_IDENTITY_MISMATCH",
                    profile_path,
                    "",
                    "profile and download manifest differ for: "
                    + ", ".join(mismatched),
                    root=catalog.spec_root,
                )
            )
    else:
        issues.append(
            _issue(
                "MODEL_DOWNLOAD_MANIFEST_TYPE_INVALID",
                manifest_path,
                "",
                "model download manifest must contain an object",
                root=catalog.spec_root,
            )
        )
    return tuple(issues)


def _extract_error_enum(openapi: Mapping[str, Any], kind: str) -> set[str]:
    schemas = openapi["components"]["schemas"]
    if kind == "top_level":
        values = schemas["Error"]["properties"]["error"]["properties"]["code"]["enum"]
    else:
        component = {
            "reason": "ReasonCode",
            "job": "JobErrorCode",
            "terminal": "TerminalReason",
        }[kind]
        values = schemas[component]["enum"]
    return set(values)


def error_bindings(
    vocabulary: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], Mapping[str, Any]], tuple[ValidationIssue, ...]]:
    """Normalize v2 code bindings and reject ambiguous multi-role rows."""

    issues: list[ValidationIssue] = []
    normalized: dict[tuple[str, str], Mapping[str, Any]] = {}
    rows = vocabulary.get("codes")
    if vocabulary.get("format") != "llm-foundations-error-vocabulary-v2":
        issues.append(
            ValidationIssue(
                "ERROR_VOCABULARY_FORMAT_INVALID",
                "contracts/error-vocabulary.json",
                "/format",
                "S0 requires the unambiguous v2 error vocabulary",
            )
        )
    if not isinstance(rows, list):
        issues.append(
            ValidationIssue(
                "ERROR_VOCABULARY_ROWS_INVALID",
                "contracts/error-vocabulary.json",
                "/codes",
                "codes must be an array",
            )
        )
        return normalized, tuple(issues)
    seen: set[str] = set()
    allowed_kinds = {"top_level", "reason", "job", "terminal"}
    binding_fields = {"http_status", "top_level_code", "retryable"}
    for index, row in enumerate(rows):
        location = f"/codes/{index}"
        if not isinstance(row, dict):
            issues.append(
                ValidationIssue(
                    "ERROR_CODE_ROW_INVALID",
                    "contracts/error-vocabulary.json",
                    location,
                    "code row must be an object",
                )
            )
            continue
        code = row.get("code")
        kinds = row.get("kinds")
        if not isinstance(code, str):
            issues.append(
                ValidationIssue(
                    "ERROR_CODE_INVALID",
                    "contracts/error-vocabulary.json",
                    f"{location}/code",
                    "code must be a string",
                )
            )
            continue
        if code in seen:
            issues.append(
                ValidationIssue(
                    "ERROR_CODE_DUPLICATE",
                    "contracts/error-vocabulary.json",
                    f"{location}/code",
                    f"duplicate error code: {code}",
                )
            )
        seen.add(code)
        if (
            not isinstance(kinds, list)
            or not kinds
            or len(kinds) != len(set(kinds))
            or not set(kinds) <= allowed_kinds
        ):
            issues.append(
                ValidationIssue(
                    "ERROR_CODE_KINDS_INVALID",
                    "contracts/error-vocabulary.json",
                    f"{location}/kinds",
                    "kinds must be a unique, non-empty subset of the closed role set",
                )
            )
            continue
        expected_code_pattern = (
            _TERMINAL_CODE_RE if set(kinds) == {"terminal"} else _CODE_RE
        )
        if not expected_code_pattern.fullmatch(code):
            issues.append(
                ValidationIssue(
                    "ERROR_CODE_INVALID",
                    "contracts/error-vocabulary.json",
                    f"{location}/code",
                    "terminal codes use lower snake case; all other roles use upper snake case",
                )
            )
            continue
        if len(kinds) == 1:
            if "bindings" in row or not binding_fields <= row.keys():
                issues.append(
                    ValidationIssue(
                        "ERROR_BINDING_SHAPE_INVALID",
                        "contracts/error-vocabulary.json",
                        location,
                        "single-role rows use exactly the flat binding fields",
                    )
                )
                continue
            role_bindings = {kinds[0]: {field: row[field] for field in binding_fields}}
        else:
            bindings = row.get("bindings")
            if any(field in row for field in binding_fields) or not isinstance(
                bindings, dict
            ) or set(bindings) != set(kinds):
                issues.append(
                    ValidationIssue(
                        "ERROR_BINDING_AMBIGUOUS",
                        "contracts/error-vocabulary.json",
                        location,
                        "multi-role rows require exact per-kind bindings and no flat binding fields",
                    )
                )
                continue
            role_bindings = bindings
        for kind in kinds:
            binding = role_bindings.get(kind)
            binding_location = (
                location
                if len(kinds) == 1
                else f"{location}/bindings/{kind}"
            )
            if not isinstance(binding, dict) or set(binding) != binding_fields:
                issues.append(
                    ValidationIssue(
                        "ERROR_BINDING_SHAPE_INVALID",
                        "contracts/error-vocabulary.json",
                        binding_location,
                        "binding must contain exactly http_status, top_level_code, retryable",
                    )
                )
                continue
            status = binding.get("http_status")
            parent = binding.get("top_level_code")
            retryable = binding.get("retryable")
            if not isinstance(retryable, bool):
                issues.append(
                    ValidationIssue(
                        "ERROR_BINDING_RETRYABLE_INVALID",
                        "contracts/error-vocabulary.json",
                        f"{binding_location}/retryable",
                        "retryable must be boolean",
                    )
                )
            if kind == "top_level":
                valid = (
                    isinstance(status, int)
                    and not isinstance(status, bool)
                    and 400 <= status <= 599
                    and parent is None
                )
            elif kind == "reason":
                valid = (
                    isinstance(status, int)
                    and not isinstance(status, bool)
                    and 400 <= status <= 599
                    and isinstance(parent, str)
                    and bool(_CODE_RE.fullmatch(parent))
                )
            else:
                valid = status is None and parent is None
            if not valid:
                issues.append(
                    ValidationIssue(
                        "ERROR_BINDING_ROLE_INVALID",
                        "contracts/error-vocabulary.json",
                        binding_location,
                        f"binding fields are invalid for role {kind}",
                    )
                )
            normalized[(code, kind)] = binding
    return normalized, tuple(issues)


def validate_semantic_and_error_bindings(
    catalog: ContractCatalog,
) -> tuple[int, int, int, tuple[ValidationIssue, ...]]:
    """Cross-check OpenAPI annotations, semantic rejections, and wire codes."""

    semantic_path = (catalog.contract_root / "semantic-rules.json").resolve()
    vocabulary_path = (catalog.contract_root / "error-vocabulary.json").resolve()
    semantic = catalog.documents[semantic_path]
    vocabulary = catalog.documents[vocabulary_path]
    issues: list[ValidationIssue] = []
    rules = semantic.get("rules") if isinstance(semantic, dict) else None
    if not isinstance(rules, list):
        return 0, 0, 0, (
            _issue(
                "SEMANTIC_RULES_INVALID",
                semantic_path,
                "/rules",
                "rules must be an array",
                root=catalog.spec_root,
            ),
        )
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict) or not isinstance(rule.get("id"), str):
            issues.append(
                _issue(
                    "SEMANTIC_RULE_INVALID",
                    semantic_path,
                    f"/rules/{index}",
                    "semantic rule must contain a string ID",
                    root=catalog.spec_root,
                )
            )
            continue
        rule_id = rule["id"]
        if rule_id in by_id:
            issues.append(
                _issue(
                    "SEMANTIC_RULE_DUPLICATE",
                    semantic_path,
                    f"/rules/{index}/id",
                    f"duplicate semantic rule: {rule_id}",
                    root=catalog.spec_root,
                )
            )
        by_id[rule_id] = rule
        if not isinstance(rule.get("applies_to"), str) or not (
            isinstance(rule.get("predicate"), str)
            or isinstance(rule.get("invariant"), str)
            or isinstance(rule.get("decision"), dict)
        ):
            issues.append(
                _issue(
                    "SEMANTIC_RULE_SHAPE_INVALID",
                    semantic_path,
                    f"/rules/{index}",
                    "rule needs applies_to and predicate, invariant, or decision",
                    root=catalog.spec_root,
                )
            )
    annotation_count = 0
    for location, annotation in _walk_key(catalog.openapi, "x-semantic-rules"):
        if (
            not isinstance(annotation, list)
            or len(annotation) != len(set(annotation))
            or not all(isinstance(item, str) for item in annotation)
        ):
            issues.append(
                _issue(
                    "SEMANTIC_ANNOTATION_INVALID",
                    catalog.openapi_path,
                    location,
                    "x-semantic-rules must be a unique string array",
                    root=catalog.spec_root,
                )
            )
            continue
        annotation_count += len(annotation)
        for rule_id in annotation:
            if rule_id not in by_id:
                issues.append(
                    _issue(
                        "SEMANTIC_RULE_UNRESOLVED",
                        catalog.openapi_path,
                        location,
                        f"x-semantic-rules names unknown rule {rule_id}",
                        root=catalog.spec_root,
                    )
                )
    normalized, binding_issues = error_bindings(vocabulary)
    issues.extend(binding_issues)
    for kind in ("top_level", "reason", "job", "terminal"):
        expected = {
            code for code, row_kind in normalized if row_kind == kind
        }
        try:
            actual = _extract_error_enum(catalog.openapi, kind)
        except (KeyError, TypeError) as exc:
            issues.append(
                _issue(
                    "OPENAPI_ERROR_ENUM_MISSING",
                    catalog.openapi_path,
                    "/components/schemas",
                    f"cannot read {kind} error enum: {exc}",
                    root=catalog.spec_root,
                )
            )
            continue
        if actual != expected:
            issues.append(
                _issue(
                    "OPENAPI_ERROR_ENUM_MISMATCH",
                    catalog.openapi_path,
                    "/components/schemas",
                    f"{kind} enum differs from error vocabulary",
                    root=catalog.spec_root,
                )
            )
    for (code, kind), binding in sorted(normalized.items()):
        if kind != "reason":
            continue
        parent_code = binding["top_level_code"]
        parent = normalized.get((parent_code, "top_level"))
        if parent is None:
            issues.append(
                _issue(
                    "ERROR_REASON_PARENT_MISSING",
                    vocabulary_path,
                    "",
                    f"reason {code} names missing top-level code {parent_code}",
                    root=catalog.spec_root,
                )
            )
        elif parent["http_status"] != binding["http_status"]:
            issues.append(
                _issue(
                    "ERROR_REASON_PARENT_CONTRADICTION",
                    vocabulary_path,
                    "",
                    f"reason {code} disagrees with top-level {parent_code}",
                    root=catalog.spec_root,
                )
            )
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict) or not isinstance(rule.get("rejection"), dict):
            continue
        rejection = rule["rejection"]
        status = rejection.get("http_status")
        code = rejection.get("code")
        reason = rejection.get("reason_code")
        top = normalized.get((code, "top_level")) if isinstance(code, str) else None
        if top is None or top.get("http_status") != status:
            issues.append(
                _issue(
                    "SEMANTIC_REJECTION_TOP_LEVEL_CONTRADICTION",
                    semantic_path,
                    f"/rules/{index}/rejection",
                    f"{rule.get('id')} rejection conflicts with top-level code {code}",
                    root=catalog.spec_root,
                )
            )
        if reason is not None:
            reason_binding = normalized.get((reason, "reason"))
            if (
                reason_binding is None
                or reason_binding.get("http_status") != status
                or reason_binding.get("top_level_code") != code
            ):
                issues.append(
                    _issue(
                        "SEMANTIC_REJECTION_REASON_CONTRADICTION",
                        semantic_path,
                        f"/rules/{index}/rejection",
                        f"{rule.get('id')} rejection conflicts with reason {reason}",
                        root=catalog.spec_root,
                    )
                )
    error_rows = vocabulary.get("codes", []) if isinstance(vocabulary, dict) else []
    return len(rules), annotation_count, len(error_rows), tuple(issues)


def validate_contract_package(spec_root: Path) -> ContractValidationReport:
    """Validate the complete S0 contract package without network or product code."""

    catalog = ContractCatalog.from_spec_root(spec_root)
    issues: list[ValidationIssue] = []
    reference_count, reference_issues = validate_reference_closure(
        catalog.contract_root,
        catalog.documents,
    )
    issues.extend(reference_issues)
    schema_count, schema_issues = validate_schema_units(catalog)
    issues.extend(schema_issues)
    operation_count, openapi_issues = validate_openapi_shape(catalog)
    issues.extend(openapi_issues)
    matrix_path = (
        catalog.spec_root / "fixtures" / "data" / "validation-cases.json"
    ).resolve()
    matrix = strict_json_load(matrix_path)
    if not isinstance(matrix, dict):
        raise ContractSourceError(
            "FIXTURE_MATRIX_TYPE_INVALID",
            "validation-cases.json must contain an object",
            source=matrix_path,
        )
    fixtures = validate_fixture_cases(catalog, matrix, matrix_path)
    issues.extend(fixtures.issues)
    model_profile_fixture_path = (
        catalog.spec_root
        / "fixtures"
        / "applied"
        / "smollm2-135m-instruct-v1.json"
    ).resolve()
    model_download_manifest_path = (
        catalog.spec_root
        / "fixtures"
        / "applied"
        / "model-download-manifest.json"
    ).resolve()
    model_profile_fixture = strict_json_load(model_profile_fixture_path)
    model_profile_issues = validate_model_profile_fixture(
        catalog,
        model_profile_fixture,
        model_profile_fixture_path,
        model_download_manifest_path,
    )
    issues.extend(model_profile_issues)
    semantic_rules, semantic_annotations, error_codes, semantic_issues = (
        validate_semantic_and_error_bindings(catalog)
    )
    issues.extend(semantic_issues)
    source_paths = set(catalog.documents)
    source_paths.add(matrix_path)
    source_paths.add((matrix_path.parent / "canonical-fixtures.json").resolve())
    source_paths.add(
        (matrix_path.parent / "materialized" / "materialized-manifest.json").resolve()
    )
    source_paths.add(model_profile_fixture_path)
    source_paths.add(model_download_manifest_path)
    source_sha256 = {
        _source_name(path, catalog.spec_root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_paths)
        if path.is_file()
    }
    return ContractValidationReport(
        spec_root=catalog.spec_root.as_posix(),
        contract_json_documents=len(catalog.documents),
        schema_units=schema_count,
        references=reference_count,
        operations=operation_count,
        fixture_cases=fixtures.declared_cases,
        fixture_cases_matched=fixtures.matched_cases,
        model_profile_fixtures=1,
        model_profile_fixtures_matched=0 if model_profile_issues else 1,
        semantic_fixture_cases=fixtures.semantic_cases,
        semantic_fixture_bindings=fixtures.semantic_binding_checks,
        materialized_fixture_files=fixtures.materialized_files,
        semantic_rules=semantic_rules,
        semantic_rule_bindings=semantic_annotations,
        error_codes=error_codes,
        source_sha256=source_sha256,
        issues=tuple(
            sorted(
                issues,
                key=lambda issue: (
                    issue.source,
                    issue.location,
                    issue.code,
                    issue.message,
                ),
            )
        ),
    )


def _error_result(exc: ContractSourceError) -> dict[str, Any]:
    return {
        "status": "ERROR",
        "issues": [
            {
                "code": exc.code,
                "source": exc.source.as_posix() if exc.source else "",
                "location": exc.location,
                "message": str(exc),
            }
        ],
        "limitations": [
            "Validation stopped at an unreadable or malformed authoritative source."
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the intermediate-v1 contract package offline."
    )
    parser.add_argument(
        "--spec-root",
        type=Path,
        required=True,
        help="path to docs/specs/intermediate-v1",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="indent the JSON report (default is one concise line)",
    )
    args = parser.parse_args(argv)
    try:
        report = validate_contract_package(args.spec_root)
        payload = report.as_dict()
        exit_code = 0 if report.ok else 1
    except ContractSourceError as exc:
        payload = _error_result(exc)
        exit_code = 2
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
