"""Authoritative JSONL dataset registration and leakage auditing."""

from __future__ import annotations

import hashlib
import io
import json
import re
import unicodedata
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, BinaryIO

from .database import utc_now
from .errors import ApiError
from .registry import Registry, StagedArtifact, _IdempotencyCommit
from .schema import canonical_json, load_document, strict_json, validate


DatasetSummary = dict[str, Any]
_SPLITS = ("train", "validation", "test")
_RECORD_FORMATS = frozenset(
    {"document_text_v1", "instruction_intent_v1", "retrieval_v1"}
)
_CODE_TOKEN_RE = re.compile(r"^[a-z]{1,3}[0-9]+$", re.ASCII)
_ASCII_DIGITS_RE = re.compile(r"[0-9]+", re.ASCII)


@dataclass(frozen=True)
class _Record:
    part: str
    line: int
    record_id: str
    scenario_group_id: str | None
    slice_name: str | None
    compared: bytes
    normalized: str
    text_bytes: int


@dataclass(frozen=True)
class _ParsedPart:
    name: str
    staged: StagedArtifact
    records: tuple[_Record, ...]
    file_bytes: int


def _field(path: str, message: str) -> dict[str, str]:
    return {"field_path": path, "message": message}


def _raise_stage(reason: str, errors: list[dict[str, str]], message: str) -> None:
    ordered = sorted(errors, key=lambda item: (item["field_path"], item["message"]))
    raise ApiError(
        "VALIDATION_FAILED",
        message,
        reason_code=reason,
        field_errors=ordered[:100],
    )


def _near_normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = "".join(
        " " if unicodedata.category(character)[:1] in {"P", "S"} else character
        for character in value
    )
    tokens = [token for token in value.split() if _CODE_TOKEN_RE.fullmatch(token) is None]
    value = _ASCII_DIGITS_RE.sub(" ", " ".join(tokens))
    return " ".join(value.split())


def _nonblank_text(value: Any) -> bool:
    return isinstance(value, str) and "\x00" not in value and bool(value.strip())


def _identifier(value: Any) -> bool:
    return _nonblank_text(value) and len(value) <= 120


def _shape_record(
    value: Any,
    *,
    record_format: str,
    part: str,
    line: int,
) -> tuple[
    _Record | None,
    list[dict[str, str]],
    list[dict[str, str]],
]:
    root = f"/{part}/{line}"
    issues: list[dict[str, str]] = []
    semantic_issues: list[dict[str, str]] = []
    if not isinstance(value, dict):
        return None, [_field(root, "Record must be a JSON object.")], []
    if record_format == "document_text_v1":
        required = {"record_id", "scenario_group_id", "text"}
        allowed = required | {"slice"}
    elif record_format == "instruction_intent_v1":
        required = {"record_id", "scenario_group_id", "messages", "expected", "slice"}
        allowed = required
    else:
        required = {"record_id", "text"}
        allowed = required
    if set(value) != allowed and not (record_format == "document_text_v1" and set(value) == required):
        issues.append(_field(root, "Record has missing or unknown fields."))
    for key in required:
        if key not in value:
            issues.append(_field(f"{root}/{key}", "Field is required."))
    if issues:
        return None, issues, []
    record_id = value.get("record_id")
    if not _identifier(record_id):
        issues.append(_field(f"{root}/record_id", "Must be a nonblank identifier of at most 120 scalars."))

    group: str | None = None
    slice_name: str | None = None
    compared_text = ""
    text_byte_count = 0
    if record_format in {"document_text_v1", "retrieval_v1"}:
        text = value.get("text")
        if not isinstance(text, str):
            issues.append(_field(f"{root}/text", "Text must be a string."))
        elif "\x00" in text or not text.strip():
            semantic_issues.append(_field(f"{root}/text", "Text must be nonblank and contain no NUL."))
        else:
            compared_text = text
            text_byte_count = len(text.encode("utf-8"))
        if record_format == "document_text_v1":
            group = value.get("scenario_group_id")
            if not _identifier(group):
                issues.append(_field(f"{root}/scenario_group_id", "Must be a nonblank identifier of at most 120 scalars."))
            if "slice" in value:
                slice_name = value.get("slice")
                if not _identifier(slice_name):
                    issues.append(_field(f"{root}/slice", "Must be a nonblank identifier of at most 120 scalars."))
    else:
        group = value.get("scenario_group_id")
        slice_name = value.get("slice")
        if not _identifier(group):
            issues.append(_field(f"{root}/scenario_group_id", "Must be a nonblank identifier of at most 120 scalars."))
        if not _identifier(slice_name):
            issues.append(_field(f"{root}/slice", "Must be a nonblank identifier of at most 120 scalars."))
        messages = value.get("messages")
        contents: list[str] = []
        if not isinstance(messages, list) or len(messages) != 1:
            issues.append(_field(f"{root}/messages", "Must contain exactly one user message."))
        else:
            message = messages[0]
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                issues.append(_field(f"{root}/messages/0", "Message must contain exactly role and content."))
            else:
                content = message.get("content")
                if not isinstance(content, str):
                    issues.append(_field(f"{root}/messages/0/content", "Message content must be a string."))
                elif message.get("role") != "user" or "\x00" in content or not content.strip():
                    semantic_issues.append(_field(f"{root}/messages/0", "Message must be a nonblank user message without NUL."))
                else:
                    contents.append(content)
        expected = value.get("expected")
        response = None
        if not isinstance(expected, dict) or set(expected) != {"intent", "response"}:
            issues.append(_field(f"{root}/expected", "Expected must contain exactly intent and response."))
        else:
            if expected.get("intent") != slice_name:
                semantic_issues.append(_field(f"{root}/expected/intent", "Expected intent must equal slice."))
            response = expected.get("response")
            if not isinstance(response, str):
                issues.append(_field(f"{root}/expected/response", "Expected response must be a string."))
            elif "\x00" in response or not response.strip():
                semantic_issues.append(_field(f"{root}/expected/response", "Expected response must be nonblank without NUL."))
            else:
                lines = response.split("\n")
                valid_response = (
                    len(lines) == 2
                    and lines[0] == f"INTENT={slice_name}"
                    and lines[1].startswith("REPLY=")
                    and bool(lines[1][6:].strip())
                    and len(lines[1][6:]) <= 240
                )
                if not valid_response:
                    semantic_issues.append(_field(f"{root}/expected/response", "Expected response is not canonical INTENT/REPLY text."))
        if contents:
            compared_text = "\n".join(contents)
            text_byte_count = sum(len(item.encode("utf-8")) for item in contents)
        if isinstance(response, str):
            text_byte_count += len(response.encode("utf-8"))
    if issues:
        return None, issues, semantic_issues
    assert isinstance(record_id, str)
    return (
        _Record(
            part,
            line,
            record_id,
            group,
            slice_name,
            compared_text.encode("utf-8"),
            _near_normalize(compared_text),
            text_byte_count,
        ),
        [],
        semantic_issues,
    )


def _cross_split_pairs(counts: Counter[str]) -> int:
    names = sorted(counts)
    return sum(counts[left] * counts[right] for index, left in enumerate(names) for right in names[index + 1 :])


def _audit_counts(records: list[_Record], record_format: str) -> tuple[int, int, int]:
    if record_format == "retrieval_v1":
        return 0, 0, 0
    exact_groups: dict[bytes, Counter[str]] = defaultdict(Counter)
    normalized_groups: dict[str, list[_Record]] = defaultdict(list)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        exact_groups[record.compared][record.part] += 1
        normalized_groups[record.normalized].append(record)
        assert record.scenario_group_id is not None
        group_splits[record.scenario_group_id].add(record.part)
    exact = sum(_cross_split_pairs(counts) for counts in exact_groups.values())
    normalized_equal = 0
    for group in normalized_groups.values():
        by_split = Counter(record.part for record in group)
        all_cross = _cross_split_pairs(by_split)
        exact_cross = 0
        exact_in_group: dict[bytes, Counter[str]] = defaultdict(Counter)
        for record in group:
            exact_in_group[record.compared][record.part] += 1
        exact_cross = sum(_cross_split_pairs(counts) for counts in exact_in_group.values())
        normalized_equal += all_cross - exact_cross
    overlap = sum(1 for splits in group_splits.values() if len(splits) > 1)
    return exact, normalized_equal, overlap


class DatasetRegistry:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self._release_families = self._load_release_families()

    @staticmethod
    def _load_release_families() -> dict[str, dict[str, str]]:
        document = load_document("materialized-manifest.json")
        rows = document.get("files")
        if document.get("format") != "llm-foundations-materialized-fixtures-v1" or not isinstance(rows, list):
            raise RuntimeError("bundled materialized fixture registry is malformed")
        paths: set[str] = set()
        families: dict[str, dict[str, str]] = defaultdict(dict)
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"path", "records", "sha256", "utf8_bytes"}:
                raise RuntimeError("bundled materialized fixture row is malformed")
            path = row.get("path")
            digest = row.get("sha256")
            records = row.get("records")
            size = row.get("utf8_bytes")
            if (
                not isinstance(path, str)
                or path in paths
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or isinstance(records, bool)
                or not isinstance(records, int)
                or records < 1
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 1
            ):
                raise RuntimeError("bundled materialized fixture row is invalid or duplicated")
            paths.add(path)
            pieces = path.split("/")
            if len(pieces) != 2:
                raise RuntimeError("bundled materialized fixture path is invalid")
            family, filename = pieces
            logical = {
                "train.jsonl": "train",
                "validation.jsonl": "validation",
                "test.jsonl": "test",
                "sealed_test.jsonl": "test",
                "documents.jsonl": "train",
            }.get(filename)
            if logical is not None:
                if logical in families[family]:
                    raise RuntimeError("bundled materialized fixture split is ambiguous")
                families[family][logical] = digest
        required = {
            "data-clinic-v1",
            "data-clinic-leaky-v1",
            "applied-intents-v1",
            "capstone-support-v1",
            "retrieval-manual-v1",
        }
        if not required.issubset(families):
            raise RuntimeError("bundled materialized fixture families are incomplete")
        return {key: dict(value) for key, value in families.items()}

    @staticmethod
    def _validate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(metadata, Mapping):
            raise TypeError("dataset metadata must be a mapping")
        value = validate("DatasetRegistrationMetadata", dict(metadata))
        if not isinstance(value, dict):
            raise RuntimeError("dataset metadata schema did not return an object")
        return value

    def _shipped_origin(
        self, record_format: str, part_digests: Mapping[str, str]
    ) -> bool:
        expected_format = {
            "data-clinic-v1": "document_text_v1",
            "data-clinic-leaky-v1": "document_text_v1",
            "applied-intents-v1": "instruction_intent_v1",
            "capstone-support-v1": "instruction_intent_v1",
            "retrieval-manual-v1": "retrieval_v1",
        }
        for family, expected in self._release_families.items():
            if expected_format.get(family) != record_format:
                continue
            if expected == dict(part_digests):
                return True
        return False

    def _parse_parts(
        self,
        staged_by_name: Mapping[str, StagedArtifact],
        record_format: str,
    ) -> list[_ParsedPart]:
        decoded: list[tuple[str, StagedArtifact, bytes, str]] = []
        encoding_errors: list[dict[str, str]] = []
        for name in _SPLITS:
            staged = staged_by_name.get(name)
            if staged is None:
                continue
            try:
                raw = staged.path.read_bytes()
            except OSError as exc:
                raise ApiError(
                    "STORAGE_UNAVAILABLE", "Staged dataset bytes are unavailable."
                ) from exc
            try:
                text = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                encoding_errors.append(
                    _field(f"/{name}", "Split file is not strict UTF-8.")
                )
                continue
            decoded.append((name, staged, raw, text))
        if encoding_errors:
            _raise_stage(
                "INVALID_ENCODING",
                encoding_errors,
                "A dataset split is not strict UTF-8.",
            )

        parsed_by_name: dict[str, list[tuple[int, Any]]] = {}
        syntax_errors: list[dict[str, str]] = []
        for name, _, _, text in decoded:
            parsed_lines: list[tuple[int, Any]] = []
            for line_number, line in enumerate(text.split("\n"), 1):
                if not line.strip():
                    continue
                try:
                    parsed_lines.append((line_number, strict_json(line)))
                except ApiError:
                    syntax_errors.append(
                        _field(
                            f"/{name}/{line_number}",
                            "Line is not one strict JSON value.",
                        )
                    )
            parsed_by_name[name] = parsed_lines
        if syntax_errors:
            _raise_stage(
                "INVALID_JSON",
                syntax_errors,
                "A dataset split contains malformed JSONL.",
            )

        parts: list[_ParsedPart] = []
        schema_errors: list[dict[str, str]] = []
        semantic_errors: list[dict[str, str]] = []
        for name, staged, raw, _ in decoded:
            records: list[_Record] = []
            parsed_lines = parsed_by_name[name]
            for line_number, value in parsed_lines:
                record, issues, semantic_issues = _shape_record(
                    value,
                    record_format=record_format,
                    part=name,
                    line=line_number,
                )
                schema_errors.extend(issues)
                semantic_errors.extend(semantic_issues)
                if record is not None:
                    records.append(record)
            if not parsed_lines:
                schema_errors.append(
                    _field(f"/{name}", "Split must contain at least one record.")
                )
            parts.append(_ParsedPart(name, staged, tuple(records), len(raw)))
        if schema_errors:
            _raise_stage(
                "SCHEMA_INVALID",
                schema_errors,
                "Dataset records do not satisfy their declared format.",
            )

        locations: dict[str, _Record] = {}
        duplicate_errors: list[dict[str, str]] = []
        for record in (record for part in parts for record in part.records):
            if record.record_id in locations:
                duplicate_errors.append(
                    _field(
                        f"/{record.part}/{record.line}/record_id",
                        "record_id is duplicated within this upload.",
                    )
                )
            else:
                locations[record.record_id] = record
        if duplicate_errors:
            _raise_stage(
                "DUPLICATE_SOURCE",
                duplicate_errors,
                "Dataset record IDs must be unique across every split.",
            )
        if semantic_errors:
            _raise_stage(
                "SEMANTIC_INVALID",
                semantic_errors,
                "Dataset records violate a semantic invariant.",
            )
        return parts

    def register(
        self,
        metadata: Mapping[str, Any],
        parts: Mapping[str, BinaryIO],
        *,
        upload_id: str | None = None,
        idempotency_commit: _IdempotencyCommit | None = None,
    ) -> DatasetSummary:
        normalized = self._validate_metadata(metadata)
        if not isinstance(parts, Mapping) or not (1 <= len(parts) <= 3):
            raise ApiError(
                "VALIDATION_FAILED",
                "Dataset upload requires one to three split files.",
                reason_code="SCHEMA_INVALID",
                field_errors=[_field("/parts", "Provide one to three split files.")],
            )
        names = set(parts)
        if names - set(_SPLITS) or any(not isinstance(name, str) for name in names):
            raise ApiError(
                "VALIDATION_FAILED",
                "Dataset split names are invalid.",
                reason_code="SCHEMA_INVALID",
                field_errors=[_field("/parts", "Only train, validation, and test are accepted.")],
            )
        record_format = str(normalized["record_format"])
        if record_format not in _RECORD_FORMATS:
            raise AssertionError("validated record format is outside the closed vocabulary")
        if record_format == "retrieval_v1" and names != {"train"}:
            _raise_stage(
                "SEMANTIC_INVALID",
                [_field("/parts", "retrieval_v1 accepts only the train part.")],
                "Retrieval datasets accept one train split only.",
            )
        owner = upload_id or str(uuid.uuid4())
        try:
            uuid.UUID(owner)
            if str(uuid.UUID(owner)) != owner:
                raise ValueError
        except (ValueError, AttributeError) as exc:
            raise ValueError("upload_id must be a lowercase UUID") from exc
        staged: list[StagedArtifact] = []
        try:
            for name in _SPLITS:
                if name not in parts:
                    continue
                staged.append(
                    self.registry.stage_stream(
                        parts[name], owner, max_bytes=10 * 1024 * 1024
                    )
                )
            if sum(item.size for item in staged) > 30 * 1024 * 1024:
                raise ApiError("PAYLOAD_TOO_LARGE", "Combined dataset split bytes exceed 30 MiB.")
            # Preserve the closed split order rather than filesystem/random order.
            staged_by_name = {name: item for name, item in zip((name for name in _SPLITS if name in parts), staged, strict=True)}
            parsed = self._parse_parts(staged_by_name, record_format)
            return self._commit_dataset(
                normalized,
                parsed,
                owner,
                idempotency_commit=idempotency_commit,
            )
        finally:
            for item in staged:
                item.path.unlink(missing_ok=True)

    def _commit_dataset(
        self,
        metadata: Mapping[str, Any],
        parsed: list[_ParsedPart],
        upload_id: str,
        *,
        idempotency_commit: _IdempotencyCommit | None,
    ) -> DatasetSummary:
        record_format = str(metadata["record_format"])
        records = [record for part in parsed for record in part.records]
        if len(records) > 100_000:
            raise ApiError("PAYLOAD_TOO_LARGE", "Dataset contains more than 100,000 records.")
        text_bytes = sum(record.text_bytes for record in records)
        if record_format == "instruction_intent_v1" and text_bytes > 20 * 1024 * 1024:
            _raise_stage(
                "PAYLOAD_TOO_LARGE",
                [_field("/parts", "Parsed instruction content exceeds 20 MiB.")],
                "Parsed instruction content is too large.",
            )
        sealed_errors: list[dict[str, str]] = []
        for part in parsed:
            if self.registry.is_sealed_digest(part.staged.sha256) and part.name != "test":
                sealed_errors.append(
                    _field(f"/{part.name}", "Sealed fixture bytes may be registered only as test.")
                )
        if sealed_errors:
            _raise_stage(
                "SEMANTIC_INVALID",
                sealed_errors,
                "Sealed fixture bytes must retain their test role.",
            )

        exact, near, overlap = _audit_counts(records, record_format)
        rejections: list[str] = []
        if exact:
            rejections.append("cross_split_exact_duplicate")
        if near:
            rejections.append("cross_split_normalized_near_duplicate")
        if overlap:
            rejections.append("group_overlap")
        eligibility = "audit_only" if rejections else "eligible"
        dataset_id = str(uuid.uuid4())
        created_at = utc_now()
        split_counts = {part.name: len(part.records) for part in parsed}
        slice_counts = Counter(
            record.slice_name for record in records if record.slice_name is not None
        )
        audit: dict[str, Any] = {
            "format": "llm-foundations-data-audit-v1",
            "dataset_id": dataset_id,
            "eligibility": eligibility,
            "record_count": len(records),
            "utf8_bytes": text_bytes,
            "split_counts": split_counts,
            "slice_counts": dict(sorted(slice_counts.items())),
            "exact_duplicate_pairs": exact,
            "normalized_near_duplicate_pairs": near,
            "group_overlap_count": overlap,
            "checked_at": created_at,
            "algorithm_version": "data-audit-v2",
            "rejections": rejections,
        }
        validate("data-audit.schema.json", audit)
        audit_artifact_id = self.registry.allocate_artifact_id()
        split_artifact_ids = {
            part.name: self.registry.allocate_artifact_id() for part in parsed
        }
        part_digests = {part.name: part.staged.sha256 for part in parsed}
        origin = (
            "shipped_fixture"
            if self._shipped_origin(record_format, part_digests)
            else "imported"
        )
        manifest_splits: dict[str, dict[str, Any]] = {}
        for part in parsed:
            manifest_splits[part.name] = {
                "artifact_id": split_artifact_ids[part.name],
                "sha256": part.staged.sha256,
                "records": len(part.records),
                "utf8_bytes": part.file_bytes,
                "sealed": self.registry.is_sealed_digest(part.staged.sha256),
            }
        content_identity = {
            "record_format": record_format,
            "splits": {
                name: {
                    "sha256": value["sha256"],
                    "records": value["records"],
                    "utf8_bytes": value["utf8_bytes"],
                    "sealed": value["sealed"],
                }
                for name, value in manifest_splits.items()
            },
        }
        manifest: DatasetSummary = {
            "format": "llm-foundations-dataset-v1",
            "dataset_id": dataset_id,
            "name": metadata["name"],
            "record_format": record_format,
            "origin": origin,
            "splits": manifest_splits,
            "audit_artifact_id": audit_artifact_id,
            "created_at": created_at,
            "manifest_sha256": hashlib.sha256(canonical_json(content_identity)).hexdigest(),
            "eligibility": eligibility,
        }
        if "provenance_note" in metadata:
            manifest["provenance_note"] = metadata["provenance_note"]
        validate("dataset-manifest.schema.json", manifest)
        audit_staged = self.registry.stage_stream(
            io.BytesIO(canonical_json(audit)),
            upload_id,
            max_bytes=1024 * 1024,
        )
        try:
            with self.registry.db.transaction() as connection:
                for part in parsed:
                    self.registry.commit_staged_in(
                        connection,
                        part.staged,
                        artifact_type="dataset_split",
                        display_name=(f"{metadata['name']} {part.name}")[:120],
                        media_type="application/x-ndjson",
                        preview_policy=(
                            "download_only"
                            if self.registry.is_sealed_digest(part.staged.sha256)
                            else "text"
                        ),
                        origin=origin,
                        artifact_id=split_artifact_ids[part.name],
                    )
                self.registry.commit_staged_in(
                    connection,
                    audit_staged,
                    artifact_type="data_audit",
                    display_name=(f"{metadata['name']} audit")[:120],
                    media_type="application/json",
                    preview_policy="text",
                    origin=origin,
                    artifact_id=audit_artifact_id,
                )
                manifest_json = canonical_json(manifest).decode("utf-8")
                connection.execute(
                    """
                    INSERT INTO datasets(
                        dataset_id, format, name, record_format, origin,
                        audit_artifact_id, created_at, provenance_note,
                        manifest_sha256, eligibility, manifest_json,
                        source_identity_json, content_revision, updated_at, deleted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, NULL)
                    """,
                    (
                        dataset_id,
                        manifest["format"],
                        manifest["name"],
                        record_format,
                        origin,
                        audit_artifact_id,
                        created_at,
                        manifest.get("provenance_note"),
                        manifest["manifest_sha256"],
                        eligibility,
                        manifest_json,
                        canonical_json(
                            {"kind": "multipart_upload", "splits": part_digests}
                        ).decode("utf-8"),
                        created_at,
                    ),
                )
                for name, value in manifest_splits.items():
                    connection.execute(
                        """
                        INSERT INTO dataset_splits(
                            dataset_id, split_name, artifact_id, sha256,
                            records, utf8_bytes, sealed
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            dataset_id,
                            name,
                            value["artifact_id"],
                            value["sha256"],
                            value["records"],
                            value["utf8_bytes"],
                            int(value["sealed"]),
                        ),
                    )
                self.registry.release_capacity(connection, upload_id)
                if idempotency_commit is not None:
                    idempotency_commit.record_success(
                        connection, status_code=201, response_body=manifest
                    )
                self.registry.db.bump_revision(connection)
            return manifest
        finally:
            audit_staged.path.unlink(missing_ok=True)

    def get(self, dataset_id: str) -> DatasetSummary:
        return self.registry.get_record("datasets", dataset_id)

    def list(
        self,
        *,
        limit: int = 50,
        cursor: str | None = None,
        record_format: str | None = None,
        origin: str | None = None,
        eligibility: str | None = None,
    ) -> dict[str, Any]:
        filters = {
            key: value
            for key, value in {
                "record_format": record_format,
                "origin": origin,
                "eligibility": eligibility,
            }.items()
            if value is not None
        }
        return self.registry.list_records(
            "datasets", limit=limit, cursor=cursor, filters=filters
        )


__all__ = ["DatasetRegistry", "DatasetSummary"]
