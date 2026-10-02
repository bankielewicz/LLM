"""Framework-free S2 admission, identity leases, and worker input snapshots."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sqlite3
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .database import Database, utc_now
from .errors import ApiError
from .platform_security import _fsync_directory, ensure_private_directory
from .registry import ROOT_QUOTA_BYTES, Registry
from .schema import canonical_json, strict_json, validate


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_U64_MAX = (1 << 64) - 1
_S2_OPERATIONS = frozenset(
    {
        "tokenizer_train",
        "tiny_train",
        "tiny_resume",
        "evaluate",
        "context_preview",
        "generate",
    }
)


@dataclass(frozen=True)
class AdmissionPlan:
    operation: str
    reservation: dict[str, int]
    resolved: dict[str, Any]
    run_id: str | None
    model_id: str | None
    tokenizer_id: str | None
    checkpoint_ids: tuple[str, ...]


@dataclass(frozen=True)
class JobContext:
    job_id: str
    operation: str
    request_sha256: str
    runtime_profile: str
    device: str
    dependency_lock_sha256: str
    companion_source_revision: str
    run_id: str | None
    model_id: str | None
    tokenizer_id: str | None
    checkpoint_ids: tuple[str, ...]
    reservation: dict[str, int]
    resolved: dict[str, Any]


@dataclass(frozen=True)
class WorkerSnapshot:
    path: Path
    sha256: str
    value: dict[str, Any]


def tiny_parameter_count(
    *, vocab_size: int, context: int, width: int, layers: int, tied: bool
) -> int:
    """Compute INT-003 unique trainable scalars without importing Torch."""
    values = (vocab_size, context, width, layers)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
        raise ValueError("tiny dimensions must be positive integers")
    count = (
        2 * vocab_size * width
        + context * width
        + layers * (12 * width * width + 13 * width)
        + 2 * width
    )
    return count - vocab_size * width if tied else count


def _checked_u64(value: int, field_path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _U64_MAX:
        raise _field_error(
            "ESTIMATE_EXCEEDS_ROOT_QUOTA",
            field_path,
            "The storage estimate cannot be represented.",
        )
    return value


def _field_error(reason: str, path: str, message: str) -> ApiError:
    return ApiError(
        "VALIDATION_FAILED",
        message,
        reason_code=reason,
        field_errors=[{"field_path": path, "message": message}],
    )


class TrainingStore:
    """Resolve accepted S2 work into immutable, framework-free execution custody."""

    def __init__(
        self,
        database: Database,
        registry: Registry,
        *,
        runtime_profile: str,
        device: str,
        dependency_lock_sha256: str,
        companion_source_revision: str | None,
        uuid_factory: Callable[[], Any] | None = None,
    ) -> None:
        if runtime_profile not in {"win-cpu", "win-cuda", "wsl-cpu", "wsl-cuda"}:
            raise ValueError("runtime_profile is invalid")
        if device != ("cuda" if runtime_profile.endswith("-cuda") else "cpu"):
            raise ValueError("device does not match runtime_profile")
        if _DIGEST.fullmatch(dependency_lock_sha256) is None:
            raise ValueError("dependency_lock_sha256 must be lowercase SHA-256")
        if companion_source_revision is not None and _REVISION.fullmatch(
            companion_source_revision
        ) is None:
            raise ValueError("companion_source_revision must be 40 lowercase hex")
        self.database = database
        self.registry = registry
        self.runtime_profile = runtime_profile
        self.device = device
        self.dependency_lock_sha256 = dependency_lock_sha256
        self.companion_source_revision = companion_source_revision
        self._uuid_factory = uuid_factory or uuid.uuid4

    def _new_id(self) -> str:
        value = str(self._uuid_factory())
        if str(uuid.UUID(value)) != value:
            raise ValueError("uuid_factory returned a noncanonical UUID")
        return value

    def _require_runtime_identity(self) -> str:
        if self.companion_source_revision is None:
            raise ApiError(
                "CAPABILITY_UNAVAILABLE",
                "S2 operations require installed build provenance.",
            )
        return self.companion_source_revision

    def _assert_request_references(self, request: Mapping[str, Any]) -> None:
        """Collect every missing client reference before any content digest work."""
        operation = request["operation"]
        wanted: list[tuple[str, str, str, str]] = []
        required_splits: tuple[str, ...] = ()
        if operation == "tokenizer_train":
            wanted.append(("datasets", "dataset_id", str(request["dataset_id"]), "/dataset_id"))
            required_splits = ("train",)
        elif operation == "tiny_train":
            wanted.extend(
                (
                    ("datasets", "dataset_id", str(request["dataset_id"]), "/dataset_id"),
                    ("tokenizers", "tokenizer_id", str(request["tokenizer_id"]), "/tokenizer_id"),
                )
            )
            required_splits = ("train", "validation")
        elif operation in {"tiny_resume", "generate"}:
            wanted.append(
                ("checkpoints", "checkpoint_id", str(request["checkpoint_id"]), "/checkpoint_id")
            )
            if operation == "generate":
                wanted.append(
                    (
                        "artifacts",
                        "artifact_id",
                        str(request["preview_artifact_id"]),
                        "/preview_artifact_id",
                    )
                )
        elif operation == "context_preview":
            if request["backend"] == "tiny":
                wanted.append(
                    (
                        "checkpoints",
                        "checkpoint_id",
                        str(request["checkpoint_id"]),
                        "/checkpoint_id",
                    )
                )
            else:
                subject = request["subject"]
                wanted.append(
                    ("models", "model_id", str(subject["model_id"]), "/subject/model_id")
                )
                if "adapter_checkpoint_id" in subject:
                    wanted.append(
                        (
                            "checkpoints",
                            "checkpoint_id",
                            str(subject["adapter_checkpoint_id"]),
                            "/subject/adapter_checkpoint_id",
                        )
                    )
        elif operation == "evaluate":
            wanted.append(("datasets", "dataset_id", str(request["dataset_id"]), "/dataset_id"))
            required_splits = (str(request["split"]),)
            for index, subject in enumerate(request["subjects"]):
                kind = subject["kind"]
                if kind == "base_model":
                    wanted.append(
                        ("models", "model_id", str(subject["model_id"]), f"/subjects/{index}/model_id")
                    )
                else:
                    wanted.append(
                        (
                            "checkpoints",
                            "checkpoint_id",
                            str(subject["checkpoint_id"]),
                            f"/subjects/{index}/checkpoint_id",
                        )
                    )

        errors: list[dict[str, str]] = []
        with self.database.read() as connection:
            for table, column, identifier, path in wanted:
                row = connection.execute(
                    f"SELECT 1 FROM {table} WHERE {column} = ? AND deleted_at IS NULL",
                    (identifier,),
                ).fetchone()
                if row is None:
                    errors.append(
                        {"field_path": path, "message": "The referenced resource does not exist."}
                    )
            dataset_id = request.get("dataset_id")
            if dataset_id is not None and required_splits and not any(
                item["field_path"] == "/dataset_id" for item in errors
            ):
                present = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT split_name FROM dataset_splits WHERE dataset_id = ?",
                        (str(dataset_id),),
                    )
                }
                if any(name not in present for name in required_splits):
                    errors.append(
                        {"field_path": "/dataset_id", "message": "The dataset is missing a required split."}
                    )
        if (
            operation == "generate"
            and len(errors) == 1
            and errors[0]["field_path"] == "/preview_artifact_id"
        ):
            raise _field_error(
                "CONTEXT_PREVIEW_STALE",
                "/preview_artifact_id",
                "The context preview no longer matches this request.",
            )
        if operation == "tiny_train" and request["architecture_profile_id"] == "tiny-v2-weight-tied-v1":
            errors.append(
                {
                    "field_path": "/e01_verification_id",
                    "message": "The required E01 verification registry is not installed.",
                }
            )
        if errors:
            raise ApiError(
                "VALIDATION_FAILED",
                "One or more referenced resources do not exist.",
                reason_code="REFERENCE_MISSING",
                field_errors=errors,
            )

    @staticmethod
    def _document_text_bytes(payload: bytes) -> int:
        total = 0
        lines = payload.splitlines()
        if not lines:
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
        for line in lines:
            try:
                value = strict_json(line)
            except (ApiError, TypeError, ValueError, UnicodeError) as exc:
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT") from exc
            if not isinstance(value, Mapping) or not isinstance(value.get("text"), str):
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
            try:
                total += len(value["text"].encode("utf-8", errors="strict"))
            except UnicodeEncodeError as exc:
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT") from exc
        return total

    def _dataset(
        self, dataset_id: str, required_splits: tuple[str, ...], *, field_path: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        with self.database.read() as connection:
            row = connection.execute(
                """
                SELECT format, record_format, eligibility, manifest_json
                FROM datasets
                WHERE dataset_id = ? AND deleted_at IS NULL
                """,
                (dataset_id,),
            ).fetchone()
            split_rows = () if row is None else tuple(
                connection.execute(
                    """
                    SELECT split_name, artifact_id, sha256, records, utf8_bytes, sealed
                    FROM dataset_splits WHERE dataset_id = ?
                    ORDER BY split_name
                    """,
                    (dataset_id,),
                )
            )
        if row is None:
            raise _field_error(
                "REFERENCE_MISSING", field_path, "The referenced dataset does not exist."
            )
        try:
            manifest = json.loads(str(row[3]))
            validate("dataset-manifest.schema.json", manifest)
        except (ApiError, TypeError, json.JSONDecodeError) as exc:
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT") from exc
        if (
            manifest["dataset_id"] != dataset_id
            or manifest["format"] != str(row[0])
            or manifest["record_format"] != str(row[1])
            or manifest["eligibility"] != str(row[2])
        ):
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
        by_name = {
            str(split[0]): {
                "dataset_id": dataset_id,
                "split": str(split[0]),
                "artifact_id": str(split[1]),
                "sha256": str(split[2]),
                "records": int(split[3]),
                "file_bytes": int(split[4]),
                "sealed": bool(split[5]),
            }
            for split in split_rows
        }
        if set(by_name) != set(manifest["splits"]):
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
        for name, split in by_name.items():
            recorded = manifest["splits"][name]
            if (
                recorded["artifact_id"] != split["artifact_id"]
                or recorded["sha256"] != split["sha256"]
                or recorded["records"] != split["records"]
                or recorded["utf8_bytes"] != split["file_bytes"]
                or recorded["sealed"] is not split["sealed"]
            ):
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
        if any(name not in by_name for name in required_splits):
            raise _field_error(
                "REFERENCE_MISSING", field_path, "The dataset is missing a required split."
            )
        selected = [by_name[name] for name in required_splits]
        for split in selected:
            descriptor = self.registry.get_artifact(split["artifact_id"])
            if (
                descriptor["sha256"] != split["sha256"]
                or descriptor["type"] != "dataset_split"
                or descriptor["size_bytes"] != split["file_bytes"]
            ):
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
            payload = self.registry.read_artifact_bytes(
                split["artifact_id"], allow_sealed_internal=split["sealed"]
            )
            split["text_bytes"] = self._document_text_bytes(payload)
            if len(payload.splitlines()) != split["records"]:
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
        return manifest, selected

    @staticmethod
    def _require_tiny_dataset(manifest: Mapping[str, Any], field_path: str) -> None:
        if (
            manifest["format"] != "llm-foundations-dataset-v1"
            or manifest["record_format"] != "document_text_v1"
        ):
            raise _field_error(
                "SUBJECT_INCOMPATIBLE",
                field_path,
                "Tiny operations require a document_text_v1 dataset.",
            )

    @staticmethod
    def _require_eligible_dataset(manifest: Mapping[str, Any], field_path: str) -> None:
        if manifest["eligibility"] != "eligible":
            raise _field_error("DATASET_INELIGIBLE", field_path, "The dataset is audit-only.")

    @staticmethod
    def _enforce_text_cap(splits: list[dict[str, Any]], names: set[str], path: str) -> None:
        total = sum(int(split["text_bytes"]) for split in splits if split["split"] in names)
        if total > 200_000:
            raise _field_error(
                "PAYLOAD_TOO_LARGE", path, "Parsed text exceeds 200,000 UTF-8 bytes."
            )

    def _input(self, role: str, artifact_id: str, *, sealed: bool = False) -> dict[str, Any]:
        descriptor = self.registry.get_artifact(artifact_id)
        self.registry.read_artifact_bytes(artifact_id, allow_sealed_internal=sealed)
        return {
            "role": role,
            "artifact_id": artifact_id,
            "sha256": descriptor["sha256"],
            "size_bytes": int(descriptor["size_bytes"]),
            "sealed": sealed,
        }

    @staticmethod
    def _tiny_tokenizer_bytes(payload: bytes) -> tuple[dict[str, Any], str, int]:
        try:
            value = strict_json(payload)
            if (
                not isinstance(value, Mapping)
                or set(value) != {"format", "merges"}
                or value["format"] != "teaching-byte-bpe-v1"
                or not isinstance(value["merges"], list)
                or len(value["merges"]) > 767
            ):
                raise ValueError
            piece_lengths = {token_id: 1 for token_id in range(256)}
            for index, merge in enumerate(value["merges"]):
                if (
                    not isinstance(merge, list)
                    or len(merge) != 2
                    or any(isinstance(item, bool) or not isinstance(item, int) for item in merge)
                ):
                    raise ValueError
                upper = 257 + index
                left, right = merge
                if any(item < 0 or item == 256 or item >= upper for item in merge):
                    raise ValueError
                size = piece_lengths[left] + piece_lengths[right]
                if size > 200_000:
                    raise ValueError
                piece_lengths[upper] = size
            normalized = dict(value)
            expected_bytes = json.dumps(normalized, indent=2).encode("utf-8")
            if payload != expected_bytes:
                raise ValueError
            fingerprint = hashlib.sha256(
                json.dumps(normalized, sort_keys=True).encode("utf-8")
            ).hexdigest()
            return normalized, fingerprint, 257 + len(value["merges"])
        except (KeyError, TypeError, ValueError, UnicodeError) as exc:
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT") from exc

    @staticmethod
    def _tiny_config_bytes(payload: bytes) -> dict[str, Any]:
        try:
            value = strict_json(payload)
            required = {
                "format",
                "architecture_profile_id",
                "vocab_size",
                "context",
                "width",
                "heads",
                "layers",
            }
            if (
                not isinstance(value, Mapping)
                or set(value) != required
                or value["format"] != "tiny-v2-config-v1"
                or value["architecture_profile_id"]
                not in {"tiny-v2-standard-v1", "tiny-v2-weight-tied-v1"}
            ):
                raise ValueError
            names = ("vocab_size", "context", "width", "heads", "layers")
            if any(
                isinstance(value[name], bool) or not isinstance(value[name], int)
                for name in names
            ):
                raise ValueError
            if not (
                257 <= value["vocab_size"] <= 1024
                and 8 <= value["context"] <= 512
                and 16 <= value["width"] <= 512
                and 1 <= value["heads"] <= 16
                and 1 <= value["layers"] <= 12
                and value["width"] % value["heads"] == 0
            ):
                raise ValueError
            tied = value["architecture_profile_id"] == "tiny-v2-weight-tied-v1"
            if tiny_parameter_count(
                vocab_size=value["vocab_size"],
                context=value["context"],
                width=value["width"],
                layers=value["layers"],
                tied=tied,
            ) > 10_000_000:
                raise ValueError
            return dict(value)
        except (KeyError, TypeError, ValueError, UnicodeError) as exc:
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT") from exc

    def _checkpoint_tiny_identity(
        self, checkpoint: Mapping[str, Any]
    ) -> tuple[dict[str, Any], str, int]:
        files = checkpoint["_files"]
        tokenizer_file = files.get("tokenizer.json")
        config_file = files.get("config.json")
        if tokenizer_file is None or config_file is None:
            raise _field_error(
                "CHECKPOINT_INCOMPATIBLE",
                "/checkpoint_id",
                "The checkpoint is missing its TinyLM identity.",
            )
        try:
            model_identity = checkpoint["model_identity"]
            if not isinstance(model_identity, Mapping):
                raise ValueError
            _, tokenizer_sha256, tokenizer_vocab_size = self._tiny_tokenizer_bytes(
                self.registry.read_artifact_bytes(tokenizer_file["artifact_id"])
            )
            config = self._tiny_config_bytes(
                self.registry.read_artifact_bytes(config_file["artifact_id"])
            )
        except (KeyError, TypeError, ValueError, UnicodeError) as exc:
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT") from exc
        if (
            config["vocab_size"] != tokenizer_vocab_size
            or model_identity.get("kind") != "tiny_v2"
            or model_identity.get("architecture_profile_id")
            != config["architecture_profile_id"]
            or model_identity.get("tokenizer_sha256") != tokenizer_sha256
            or model_identity.get("config_sha256") != config_file["sha256"]
        ):
            raise _field_error(
                "CHECKPOINT_INCOMPATIBLE",
                "/checkpoint_id",
                "The checkpoint declares incompatible TinyLM identities.",
            )
        return config, tokenizer_sha256, tokenizer_vocab_size

    def _tokenizer(self, tokenizer_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        with self.database.read() as connection:
            row = connection.execute(
                """
                SELECT tokenizer_type, vocab_size, tokenizer_sha256, artifact_id,
                       created_at, record_json
                FROM tokenizers
                WHERE tokenizer_id = ? AND deleted_at IS NULL
                """,
                (tokenizer_id,),
            ).fetchone()
        if row is None:
            raise _field_error(
                "REFERENCE_MISSING", "/tokenizer_id", "The tokenizer does not exist."
            )
        artifact_id = str(row[3])
        try:
            record = strict_json(str(row[5]))
            validate("TokenizerDetail", record)
            if (
                not isinstance(record, Mapping)
                or record["tokenizer_id"] != tokenizer_id
                or record["tokenizer_type"] != str(row[0])
                or record["vocab_size"] != int(row[1])
                or record["tokenizer_sha256"] != str(row[2])
                or record["created_at"] != str(row[4])
                or record["artifact_ids"] != [artifact_id]
            ):
                raise ValueError
        except (ApiError, TypeError, ValueError, UnicodeError) as exc:
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT") from exc
        payload = self.registry.read_artifact_bytes(artifact_id)
        value, fingerprint, vocab_size = self._tiny_tokenizer_bytes(payload)
        if (
            fingerprint != str(row[2])
            or int(row[1]) != vocab_size
        ):
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
        return (
            {
                "tokenizer_id": tokenizer_id,
                "tokenizer_type": str(row[0]),
                "vocab_size": int(row[1]),
                "tokenizer_sha256": str(row[2]),
                "artifact_id": artifact_id,
            },
            self._input("tokenizer_json", artifact_id),
        )

    def _checkpoint(
        self, checkpoint_id: str, *, require_resume: bool = False
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        with self.database.read() as connection:
            row = connection.execute(
                """
                SELECT format, backend, origin, run_id, job_id,
                       manifest_sha256, record_json
                FROM checkpoints
                WHERE checkpoint_id = ? AND deleted_at IS NULL
                """,
                (checkpoint_id,),
            ).fetchone()
            files = () if row is None else tuple(
                connection.execute(
                    """
                    SELECT file_name, artifact_id, sha256, size_bytes
                    FROM checkpoint_files WHERE checkpoint_id = ?
                    ORDER BY file_name
                    """,
                    (checkpoint_id,),
                )
            )
        if row is None:
            raise _field_error(
                "REFERENCE_MISSING", "/checkpoint_id", "The checkpoint does not exist."
            )
        try:
            descriptor = json.loads(str(row[6]))
            validate("checkpoint-descriptor.schema.json", descriptor)
        except (ApiError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT") from exc
        if (
            descriptor["checkpoint_id"] != checkpoint_id
            or descriptor["format"] != str(row[0])
            or descriptor["backend"] != str(row[1])
            or descriptor["origin"] != str(row[2])
            or descriptor["run_id"] != str(row[3])
            or descriptor["job_id"] != str(row[4])
            or descriptor["sha256"] != str(row[5])
        ):
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
        inputs: list[dict[str, Any]] = []
        file_map: dict[str, dict[str, Any]] = {}
        for file_name, artifact_id, digest, size_bytes in files:
            item = self._input(f"checkpoint.{file_name}", str(artifact_id))
            if item["sha256"] != str(digest) or item["size_bytes"] != int(size_bytes):
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
            inputs.append(item)
            file_map[str(file_name)] = item
        manifest_item = file_map.get("manifest.json")
        if (
            manifest_item is None
            or manifest_item["artifact_id"] != descriptor.get("artifact_id")
            or manifest_item["sha256"] != descriptor.get("sha256")
        ):
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
        if descriptor.get("backend") != "tiny_v2":
            raise _field_error(
                "SUBJECT_INCOMPATIBLE",
                "/checkpoint_id",
                "The checkpoint is not a TinyLM checkpoint.",
            )

        manifest_raw = self.registry.read_artifact_bytes(manifest_item["artifact_id"])
        try:
            manifest_value = strict_json(manifest_raw)
            validate("checkpoint-manifest.schema.json", manifest_value)
            if not isinstance(manifest_value, Mapping):
                raise ValueError
            manifest_files: dict[str, Mapping[str, Any]] = {}
            for entry in manifest_value["files"]:
                if not isinstance(entry, Mapping):
                    raise ValueError
                name = entry["name"]
                if not isinstance(name, str) or name in manifest_files:
                    raise ValueError
                manifest_files[name] = entry
            model_identity = descriptor["model_identity"]
            if not isinstance(model_identity, Mapping):
                raise ValueError
            expected_aliases = (
                {"output.weight": "tokens.weight"}
                if model_identity.get("architecture_profile_id")
                == "tiny-v2-weight-tied-v1"
                else {}
            )
            if (
                manifest_value["format"] != "tiny-v2-checkpoint-v1"
                or manifest_value["portability"] != descriptor["portability"]
                or manifest_value["aliases"] != expected_aliases
                or set(file_map) != set(manifest_files) | {"manifest.json"}
            ):
                raise ValueError
            for name, entry in manifest_files.items():
                item = file_map[name]
                if (
                    entry["sha256"] != item["sha256"]
                    or entry["size_bytes"] != item["size_bytes"]
                ):
                    raise ValueError
        except (ApiError, KeyError, TypeError, ValueError, UnicodeError) as exc:
            raise _field_error(
                "CHECKPOINT_INCOMPATIBLE",
                "/checkpoint_id",
                "The checkpoint manifest is incompatible.",
            ) from exc
        if require_resume and (
            descriptor.get("portability") != "resume"
            or descriptor.get("contains_resume_state") is not True
        ):
            raise _field_error(
                "CHECKPOINT_NOT_RESUMABLE",
                "/checkpoint_id",
                "The checkpoint does not contain resume state.",
            )
        descriptor["_files"] = file_map
        return descriptor, inputs

    def _checkpoint_resolution(
        self, checkpoint_id: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        checkpoint, inputs = self._checkpoint(checkpoint_id)
        config, tokenizer_sha256, _ = self._checkpoint_tiny_identity(checkpoint)
        model_identity = checkpoint["model_identity"]

        with self.database.read() as connection:
            context_row = connection.execute(
                """
                SELECT model_id FROM job_operation_contexts
                WHERE job_id = ?
                """,
                (checkpoint["job_id"],),
            ).fetchone()
            model_rows = tuple(
                connection.execute(
                    "SELECT model_id, record_json FROM models WHERE deleted_at IS NULL"
                )
            )
        model_id = str(context_row[0]) if context_row is not None else None
        registered: list[str] = []
        for row_model_id, record_json in model_rows:
            try:
                record = json.loads(str(record_json))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT") from exc
            if isinstance(record, Mapping) and record.get("checkpoint_id") == checkpoint_id:
                if record.get("checkpoint_sha256") != checkpoint["sha256"]:
                    raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
                registered.append(str(row_model_id))
        if registered:
            if len(registered) != 1 or (
                model_id is not None and registered[0] != model_id
            ):
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
            model_id = registered[0]
        if model_id is None:
            raise _field_error(
                "REFERENCE_MISSING",
                "/checkpoint_id",
                "The checkpoint has no resolved model identity.",
            )
        resolved = {
            "checkpoint_id": checkpoint_id,
            "model_id": model_id,
            "manifest_sha256": checkpoint["sha256"],
            "checkpoint_sha256": checkpoint["sha256"],
            "tokenizer_sha256": tokenizer_sha256,
            "config_sha256": checkpoint["_files"]["config.json"]["sha256"],
            "context": config["context"],
            "architecture_profile_id": model_identity["architecture_profile_id"],
            "runtime_profile": self.runtime_profile,
            "device": self.device,
        }
        return resolved, inputs

    def _tiny_reservation(
        self, *, parameter_count: int, steps: int, eval_every: int, field_path: str
    ) -> tuple[dict[str, int], int]:
        boundaries = _checked_u64(2 + (steps + eval_every - 1) // eval_every, field_path)
        per_boundary = _checked_u64(12 * parameter_count + 8 * 1024 * 1024, field_path)
        estimate = _checked_u64(boundaries * per_boundary + 16 * 1024 * 1024, field_path)
        estimate = max(512 * 1024 * 1024, estimate)
        if estimate > ROOT_QUOTA_BYTES:
            raise _field_error(
                "ESTIMATE_EXCEEDS_ROOT_QUOTA",
                field_path,
                "The requested run exceeds the storage-root quota.",
            )
        return (
            {
                "byte_count": estimate,
                "artifact_rows": 7 * boundaries + 8,
                "dataset_rows": 0,
                "run_rows": 1,
                "model_rows": 1,
                "checkpoint_rows": boundaries,
            },
            boundaries,
        )

    @staticmethod
    def _ordinary_reservation(*, run_rows: int = 0, model_rows: int = 0) -> dict[str, int]:
        return {
            "byte_count": 256 * 1024 * 1024,
            "artifact_rows": 16,
            "dataset_rows": 0,
            "run_rows": run_rows,
            "model_rows": model_rows,
            "checkpoint_rows": 0,
        }

    def resolve(self, request: Mapping[str, Any]) -> AdmissionPlan:
        self._require_runtime_identity()
        operation = request.get("operation")
        if operation not in _S2_OPERATIONS:
            raise ApiError("CAPABILITY_UNAVAILABLE")
        value = dict(validate("JobRequest", dict(request), include_semantic=False))
        self._assert_request_references(value)
        if operation == "tokenizer_train":
            manifest, splits = self._dataset(
                str(value["dataset_id"]), ("train",), field_path="/dataset_id"
            )
            validate("JobRequest", value)
            self._require_tiny_dataset(manifest, "/dataset_id")
            self._enforce_text_cap(splits, {"train"}, "/dataset_id")
            self._require_eligible_dataset(manifest, "/dataset_id")
            inputs = [
                self._input(
                    "dataset.train", split["artifact_id"], sealed=bool(split["sealed"])
                )
                for split in splits
            ]
            return AdmissionPlan(
                operation=operation,
                reservation={
                    "byte_count": 64 * 1024 * 1024,
                    "artifact_rows": 16,
                    "dataset_rows": 0,
                    "run_rows": 1,
                    "model_rows": 0,
                    "checkpoint_rows": 0,
                },
                resolved={
                    "request": value,
                    "dataset_manifest_sha256": manifest["manifest_sha256"],
                    "dataset_bindings": splits,
                    "_inputs": inputs,
                },
                run_id=self._new_id(),
                model_id=None,
                tokenizer_id=self._new_id(),
                checkpoint_ids=(),
            )
        if operation == "tiny_train":
            manifest, splits = self._dataset(
                str(value["dataset_id"]),
                ("train", "validation"),
                field_path="/dataset_id",
            )
            tokenizer, tokenizer_input = self._tokenizer(str(value["tokenizer_id"]))
            validate("JobRequest", value)
            self._require_tiny_dataset(manifest, "/dataset_id")
            self._enforce_text_cap(splits, {"train"}, "/dataset_id")
            tied = value["architecture_profile_id"] == "tiny-v2-weight-tied-v1"
            parameters = tiny_parameter_count(
                vocab_size=tokenizer["vocab_size"],
                context=int(value["context"]),
                width=int(value["width"]),
                layers=int(value["layers"]),
                tied=tied,
            )
            if parameters > 10_000_000:
                raise _field_error(
                    "SEMANTIC_INVALID", "/width", "Unique parameters exceed 10,000,000."
                )
            self._require_eligible_dataset(manifest, "/dataset_id")
            reservation, boundaries = self._tiny_reservation(
                parameter_count=parameters,
                steps=int(value["steps"]),
                eval_every=int(value["eval_every"]),
                field_path="/eval_every",
            )
            inputs = [
                self._input(
                    f"dataset.{split['split']}",
                    split["artifact_id"],
                    sealed=bool(split["sealed"]),
                )
                for split in splits
            ]
            inputs.append(tokenizer_input)
            return AdmissionPlan(
                operation=operation,
                reservation=reservation,
                resolved={
                    "request": value,
                    "dataset_manifest_sha256": manifest["manifest_sha256"],
                    "dataset_bindings": splits,
                    "tokenizer": tokenizer,
                    "parameter_count": parameters,
                    "requested_final_step": int(value["steps"]),
                    "_inputs": inputs,
                },
                run_id=self._new_id(),
                model_id=self._new_id(),
                tokenizer_id=None,
                checkpoint_ids=tuple(self._new_id() for _ in range(boundaries)),
            )
        if operation == "tiny_resume":
            checkpoint, checkpoint_inputs = self._checkpoint(
                str(value["checkpoint_id"]), require_resume=True
            )
            trainer_item = checkpoint["_files"]["trainer_state.json"]
            try:
                trainer = strict_json(
                    self.registry.read_artifact_bytes(trainer_item["artifact_id"])
                )
                validate("tiny-trainer-state.schema.json", trainer)
            except (ApiError, TypeError, ValueError, UnicodeError) as exc:
                raise _field_error(
                    "CHECKPOINT_INCOMPATIBLE",
                    "/checkpoint_id",
                    "The checkpoint trainer state is incompatible.",
                ) from exc
            validate("JobRequest", value)
            if (
                trainer["runtime_profile"] != self.runtime_profile
                or trainer["device"] != self.device
                or trainer["dependency_lock_sha256"] != self.dependency_lock_sha256
                or trainer["completed_global_step"] != checkpoint["step"]
                or trainer["seed"] != checkpoint["seed"]
                or trainer["tokenizer_sha256"]
                != checkpoint["model_identity"]["tokenizer_sha256"]
                or trainer["architecture_profile_id"]
                != checkpoint["model_identity"]["architecture_profile_id"]
            ):
                raise _field_error(
                    "CHECKPOINT_INCOMPATIBLE",
                    "/checkpoint_id",
                    "The checkpoint runtime identity is incompatible.",
                )
            final_step = int(checkpoint["step"]) + int(value["additional_steps"])
            if final_step > 2_147_483_647:
                raise _field_error(
                    "TINY_STEP_OVERFLOW",
                    "/additional_steps",
                    "The resumed global step would overflow.",
                )
            training = checkpoint["training_identity"]
            config, tokenizer_sha256, vocab_size = self._checkpoint_tiny_identity(
                checkpoint
            )
            if (
                any(
                    trainer[name] != training[name]
                    for name in ("batch_size", "learning_rate", "eval_every")
                )
                or training["save_every"] != training["eval_every"]
                or any(
                    config[name] != training[name]
                    for name in ("context", "width", "heads", "layers")
                )
            ):
                raise _field_error(
                    "CHECKPOINT_INCOMPATIBLE",
                    "/checkpoint_id",
                    "The checkpoint configuration identities are incompatible.",
                )
            parameters = tiny_parameter_count(
                vocab_size=vocab_size,
                context=int(training["context"]),
                width=int(training["width"]),
                layers=int(training["layers"]),
                tied=checkpoint["model_identity"]["architecture_profile_id"]
                == "tiny-v2-weight-tied-v1",
            )
            reservation, boundaries = self._tiny_reservation(
                parameter_count=parameters,
                steps=int(value["additional_steps"]),
                eval_every=int(training["eval_every"]),
                field_path="/additional_steps",
            )
            checkpoint_bindings = {
                str(binding["split"]): dict(binding)
                for binding in checkpoint["dataset_bindings"]
            }
            trainer_bindings = {
                str(binding["split"]): dict(binding)
                for binding in trainer["dataset_bindings"]
            }
            dataset_ids = {
                str(binding["dataset_id"]) for binding in checkpoint_bindings.values()
            }
            if (
                set(checkpoint_bindings) != {"train", "validation"}
                or set(trainer_bindings) != {"train", "validation"}
                or len(dataset_ids) != 1
            ):
                raise _field_error(
                    "CHECKPOINT_INCOMPATIBLE",
                    "/checkpoint_id",
                    "The checkpoint dataset identities are incompatible.",
                )
            current_manifest, current_splits = self._dataset(
                next(iter(dataset_ids)),
                ("train", "validation"),
                field_path="/checkpoint_id",
            )
            self._require_tiny_dataset(current_manifest, "/checkpoint_id")
            self._enforce_text_cap(current_splits, {"train"}, "/checkpoint_id")
            dataset_inputs: list[dict[str, Any]] = []
            bindings: list[dict[str, Any]] = []
            for split in current_splits:
                name = str(split["split"])
                checkpoint_binding = checkpoint_bindings[name]
                trainer_binding = trainer_bindings[name]
                if (
                    checkpoint_binding["dataset_id"] != split["dataset_id"]
                    or checkpoint_binding["artifact_id"] != split["artifact_id"]
                    or checkpoint_binding["sha256"] != split["sha256"]
                    or trainer_binding
                    != {
                        "split": name,
                        "dataset_id": split["dataset_id"],
                        "sha256": split["sha256"],
                    }
                ):
                    raise _field_error(
                        "CHECKPOINT_INCOMPATIBLE",
                        "/checkpoint_id",
                        "The checkpoint dataset identity changed.",
                    )
                dataset_inputs.append(
                    self._input(f"dataset.{name}", split["artifact_id"])
                )
                bindings.append(checkpoint_binding)
            self._require_eligible_dataset(current_manifest, "/checkpoint_id")
            return AdmissionPlan(
                operation=operation,
                reservation=reservation,
                resolved={
                    "request": value,
                    "parent_checkpoint": {
                        key: item for key, item in checkpoint.items() if key != "_files"
                    },
                    "dataset_bindings": bindings,
                    "training_identity": dict(training),
                    "parameter_count": parameters,
                    "requested_final_step": final_step,
                    "_inputs": checkpoint_inputs + dataset_inputs,
                },
                run_id=self._new_id(),
                model_id=self._new_id(),
                tokenizer_id=None,
                checkpoint_ids=tuple(self._new_id() for _ in range(boundaries)),
            )
        if operation == "evaluate":
            manifest, splits = self._dataset(
                str(value["dataset_id"]),
                (str(value["split"]),),
                field_path="/dataset_id",
            )
            subject_values: list[dict[str, Any]] = []
            inputs: list[dict[str, Any]] = []
            for index, subject in enumerate(value["subjects"]):
                if subject["kind"] != "tiny_checkpoint":
                    continue
                resolved, checkpoint_inputs = self._checkpoint_resolution(
                    str(subject["checkpoint_id"])
                )
                subject_values.append({"checkpoint": resolved})
                for item in checkpoint_inputs:
                    copied = dict(item)
                    copied["role"] = f"subject.{index}.{item['role']}"
                    inputs.append(copied)
            validate("JobRequest", value)
            if value["evaluation_profile_id"] != "tiny-nll-per-byte-v1":
                raise _field_error(
                    "SUBJECT_INCOMPATIBLE",
                    "/evaluation_profile_id",
                    "This runtime slice evaluates only TinyLM checkpoints.",
                )
            self._require_tiny_dataset(manifest, "/dataset_id")
            self._enforce_text_cap(splits, {str(value["split"])}, "/dataset_id")
            self._require_eligible_dataset(manifest, "/dataset_id")
            if any(bool(split["sealed"]) for split in splits):
                raise ApiError(
                    "STATE_CONFLICT",
                    "Sealed evaluation requires a consumed release token.",
                    reason_code="SEALED_SPLIT_REQUIRES_TOKEN",
                )
            inputs[0:0] = [
                self._input(
                    f"dataset.{split['split']}",
                    split["artifact_id"],
                    sealed=False,
                )
                for split in splits
            ]
            run_id = self._new_id()
            return AdmissionPlan(
                operation=operation,
                reservation=self._ordinary_reservation(run_rows=1),
                resolved={
                    "request": value,
                    "evaluation_id": run_id,
                    "dataset_manifest_sha256": manifest["manifest_sha256"],
                    "dataset_bindings": splits,
                    "subjects": subject_values,
                    "_inputs": inputs,
                },
                run_id=run_id,
                model_id=None,
                tokenizer_id=None,
                checkpoint_ids=(),
            )
        if operation == "context_preview":
            if value["backend"] != "tiny":
                validate("JobRequest", value)
                raise ApiError(
                    "CAPABILITY_UNAVAILABLE",
                    "Applied-model context preview is not available in this runtime slice.",
                )
            checkpoint, checkpoint_inputs = self._checkpoint_resolution(
                str(value["checkpoint_id"])
            )
            validate("JobRequest", value)
            wanted = {"checkpoint.tokenizer.json", "checkpoint.config.json"}
            inputs = [item for item in checkpoint_inputs if item["role"] in wanted]
            if {item["role"] for item in inputs} != wanted:
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
            return AdmissionPlan(
                operation=operation,
                reservation=self._ordinary_reservation(),
                resolved={
                    "request": value,
                    "checkpoint": checkpoint,
                    "_inputs": inputs,
                },
                run_id=None,
                model_id=None,
                tokenizer_id=None,
                checkpoint_ids=(),
            )
        if operation == "generate":
            checkpoint, checkpoint_inputs = self._checkpoint_resolution(
                str(value["checkpoint_id"])
            )
            preview_id = str(value["preview_artifact_id"])
            try:
                preview_descriptor = self.registry.get_artifact(preview_id)
                if preview_descriptor.get("type") != "context_preview":
                    raise _field_error(
                        "CONTEXT_PREVIEW_STALE",
                        "/preview_artifact_id",
                        "The context preview no longer matches this request.",
                    )
                preview_bytes = self.registry.read_artifact_bytes(preview_id)
            except ApiError as exc:
                if exc.code != "NOT_FOUND":
                    raise
                raise _field_error(
                    "CONTEXT_PREVIEW_STALE",
                    "/preview_artifact_id",
                    "The context preview no longer matches this request.",
                ) from exc
            try:
                preview = strict_json(preview_bytes)
                if not isinstance(preview, Mapping):
                    raise ValueError
                preview_value = dict(preview)
                expected_preview_keys = {
                    "backend",
                    "subject_identity_sha256",
                    "tokenizer_sha256",
                    "template_sha256",
                    "device",
                    "original_input_token_count",
                    "input_token_ids",
                    "input_token_count",
                    "serialized_text",
                    "dropped_message_indices",
                    "cropped_input_tokens",
                    "effective_context_budget",
                    "canonical_generation_request_sha256",
                }
                if set(preview_value) != expected_preview_keys:
                    raise ValueError
                preview_digest = hashlib.sha256(canonical_json(preview_value)).hexdigest()
                preview_result = {
                    "operation": "context_preview",
                    "preview_artifact_id": preview_id,
                    **preview_value,
                    "context_preview_digest": preview_digest,
                    "artifact_ids": [preview_id],
                }
                validate("ContextPreviewResult", preview_result)
                generation_request = {
                    key: item
                    for key, item in value.items()
                    if key not in {"preview_artifact_id", "context_preview_digest"}
                }
                subject_identity = {
                    "backend": "tiny",
                    "model_id": checkpoint["model_id"],
                    "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                    "runtime_profile": self.runtime_profile,
                    "device": self.device,
                    "context_limit": checkpoint["context"],
                }
                subject_sha256 = hashlib.sha256(
                    canonical_json(subject_identity)
                ).hexdigest()
                if (
                    preview_bytes != canonical_json(preview_value)
                    or preview_descriptor["sha256"] != preview_digest
                    or
                    value["context_preview_digest"] != preview_digest
                    or preview_value["canonical_generation_request_sha256"]
                    != hashlib.sha256(canonical_json(generation_request)).hexdigest()
                    or preview_value["subject_identity_sha256"] != subject_sha256
                    or preview_value["tokenizer_sha256"]
                    != checkpoint["tokenizer_sha256"]
                    or preview_value["template_sha256"] is not None
                    or preview_value["device"] != self.device
                    or preview_value["effective_context_budget"]
                    != checkpoint["context"]
                ):
                    raise ValueError
            except ApiError as exc:
                raise _field_error(
                    "CONTEXT_PREVIEW_STALE",
                    "/preview_artifact_id",
                    "The context preview no longer matches this request.",
                ) from exc
            except (KeyError, TypeError, ValueError, UnicodeError) as exc:
                raise _field_error(
                    "CONTEXT_PREVIEW_STALE",
                    "/preview_artifact_id",
                    "The context preview no longer matches this request.",
                ) from exc
            validate("JobRequest", value)
            try:
                preview_input = self._input("context_preview", preview_id)
            except ApiError as exc:
                if exc.code != "NOT_FOUND":
                    raise
                raise _field_error(
                    "CONTEXT_PREVIEW_STALE",
                    "/preview_artifact_id",
                    "The context preview no longer matches this request.",
                ) from exc
            return AdmissionPlan(
                operation=operation,
                reservation=self._ordinary_reservation(run_rows=1),
                resolved={
                    "request": value,
                    "checkpoint": checkpoint,
                    "subject_identity": subject_identity,
                    "preview": preview_value,
                    "_inputs": checkpoint_inputs + [preview_input],
                },
                run_id=self._new_id(),
                model_id=None,
                tokenizer_id=None,
                checkpoint_ids=(),
            )
        raise ApiError("CAPABILITY_UNAVAILABLE")

    def create_job_context_in(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        request_sha256: str,
        plan: AdmissionPlan,
    ) -> None:
        revision = self._require_runtime_identity()
        if plan.operation not in _S2_OPERATIONS or _DIGEST.fullmatch(request_sha256) is None:
            raise ValueError("invalid admitted operation context")
        connection.execute(
            """
            INSERT INTO job_operation_contexts(
                job_id, operation, request_sha256, runtime_profile, device,
                dependency_lock_sha256, companion_source_revision, run_id,
                model_id, tokenizer_id, checkpoint_ids_json, reservation_json,
                resolved_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                plan.operation,
                request_sha256,
                self.runtime_profile,
                self.device,
                self.dependency_lock_sha256,
                revision,
                plan.run_id,
                plan.model_id,
                plan.tokenizer_id,
                canonical_json(list(plan.checkpoint_ids)).decode("utf-8"),
                canonical_json(plan.reservation).decode("utf-8"),
                canonical_json(plan.resolved).decode("utf-8"),
                utc_now(),
            ),
        )

    def get_job_context(self, job_id: str) -> JobContext:
        with self.database.read() as connection:
            row = connection.execute(
                """
                SELECT job_id, operation, request_sha256, runtime_profile, device,
                       dependency_lock_sha256, companion_source_revision, run_id,
                       model_id, tokenizer_id, checkpoint_ids_json,
                       reservation_json, resolved_json
                FROM job_operation_contexts WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            raise ApiError("NOT_FOUND", "The job operation context was not found.")
        try:
            checkpoint_ids = strict_json(str(row[10]))
            reservation = strict_json(str(row[11]))
            resolved = strict_json(str(row[12]))
            if (
                not isinstance(checkpoint_ids, list)
                or not isinstance(reservation, Mapping)
                or not isinstance(resolved, Mapping)
            ):
                raise ValueError
        except (TypeError, ValueError, UnicodeError) as exc:
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT") from exc
        return JobContext(
            job_id=str(row[0]),
            operation=str(row[1]),
            request_sha256=str(row[2]),
            runtime_profile=str(row[3]),
            device=str(row[4]),
            dependency_lock_sha256=str(row[5]),
            companion_source_revision=str(row[6]),
            run_id=None if row[7] is None else str(row[7]),
            model_id=None if row[8] is None else str(row[8]),
            tokenizer_id=None if row[9] is None else str(row[9]),
            checkpoint_ids=tuple(str(value) for value in checkpoint_ids),
            reservation={str(key): int(value) for key, value in reservation.items()},
            resolved=dict(resolved),
        )

    @staticmethod
    def _write_verified_copy(
        destination: Path,
        source: Any,
        *,
        expected_size: int,
        expected_sha256: str,
    ) -> None:
        if destination.exists():
            info = os.lstat(destination)
            if not stat.S_ISREG(info.st_mode) or info.st_size != expected_size:
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
            digest = hashlib.sha256()
            with destination.open("rb") as current:
                for chunk in iter(lambda: current.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
            return
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4()}.partial")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb", buffering=0) as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            info = os.lstat(temporary)
            if not stat.S_ISREG(info.st_mode) or info.st_size != expected_size:
                raise OSError("materialized input size mismatch")
            digest = hashlib.sha256()
            with temporary.open("rb") as copied:
                for chunk in iter(lambda: copied.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise OSError("materialized input digest mismatch")
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def materialize_worker_snapshot(self, job_id: str) -> WorkerSnapshot:
        context = self.get_job_context(job_id)
        raw_inputs = context.resolved.get("_inputs")
        if not isinstance(raw_inputs, list):
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
        staging = self.registry.create_staging_dir(job_id)
        inputs_root = staging / "inputs"
        ensure_private_directory(inputs_root)
        materialized: list[dict[str, Any]] = []
        roles: set[str] = set()
        for index, raw in enumerate(raw_inputs):
            if not isinstance(raw, Mapping):
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
            role = str(raw.get("role"))
            if role in roles or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", role) is None:
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
            roles.add(role)
            artifact_id = str(raw.get("artifact_id"))
            digest = str(raw.get("sha256"))
            size = raw.get("size_bytes")
            sealed = raw.get("sealed", False)
            if (
                _DIGEST.fullmatch(digest) is None
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or not isinstance(sealed, bool)
            ):
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
            name = f"{index:02d}-{role}"
            destination = inputs_root / name
            with self.registry.open_verified_artifact(
                artifact_id, allow_sealed_internal=sealed
            ) as source:
                self._write_verified_copy(
                    destination,
                    source,
                    expected_size=size,
                    expected_sha256=digest,
                )
            materialized.append(
                {
                    "role": role,
                    "artifact_id": artifact_id,
                    "sha256": digest,
                    "size_bytes": size,
                    "path": f"inputs/{name}",
                }
            )
        resolved = {
            key: value for key, value in context.resolved.items() if key != "_inputs"
        }
        snapshot_value = {
            "format": "llm-foundations-worker-input-v1",
            "job_id": context.job_id,
            "operation": context.operation,
            "request_sha256": context.request_sha256,
            "runtime_profile": context.runtime_profile,
            "device": context.device,
            "dependency_lock_sha256": context.dependency_lock_sha256,
            "companion_source_revision": context.companion_source_revision,
            "ids": {
                "run_id": context.run_id,
                "model_id": context.model_id,
                "tokenizer_id": context.tokenizer_id,
                "checkpoint_ids": list(context.checkpoint_ids),
            },
            "resolved": resolved,
            "inputs": materialized,
        }
        snapshot_bytes = canonical_json(snapshot_value)
        snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
        snapshot_path = staging.parent / "input-snapshot.json"
        self._write_verified_copy(
            snapshot_path,
            source=io.BytesIO(snapshot_bytes),
            expected_size=len(snapshot_bytes),
            expected_sha256=snapshot_sha256,
        )
        return WorkerSnapshot(snapshot_path, snapshot_sha256, snapshot_value)


__all__ = [
    "AdmissionPlan",
    "JobContext",
    "TrainingStore",
    "WorkerSnapshot",
    "tiny_parameter_count",
]
