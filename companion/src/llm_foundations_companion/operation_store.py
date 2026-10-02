"""Atomic custody for staged S2 worker artifacts and TinyLM checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .database import Database, utc_now
from .errors import ApiError
from .platform_security import _reject_linked_components, is_reparse_point
from .registry import FREE_SPACE_RESERVE_BYTES, Registry, StagedArtifact
from .schema import canonical_json, strict_json, validate
from .tiny_artifact_formats import validate_tiny_paired_evaluation
from .tiny_evaluation_records import (
    expected_tiny_evaluation_records,
    validate_tiny_evaluation_records,
)
from .tiny_v2.tokenizer import (
    load_tokenizer_bytes,
    serialize_tokenizer,
    tokenizer_type,
)
from .training_store import JobContext, TrainingStore


_CHECKPOINT_NAMES = (
    "model.safetensors",
    "optimizer.safetensors",
    "rng.safetensors",
    "tokenizer.json",
    "config.json",
    "trainer_state.json",
    "manifest.json",
)
_MANIFEST_PAYLOAD_NAMES = frozenset(_CHECKPOINT_NAMES[:-1])
_TRAINING_OPERATIONS = frozenset({"tiny_train", "tiny_resume"})
_TERMINAL_STATES = frozenset({"completed", "failed", "interrupted"})
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_BASENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")


class OperationStoreError(ValueError):
    """A worker proposal that maps to a closed job error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PreparedArtifact:
    job_id: str
    role: str
    staging_name: str
    artifact_id: str
    staged: StagedArtifact


@dataclass(frozen=True)
class ArtifactCommit:
    ack: dict[str, Any]


@dataclass(frozen=True)
class PreparedCheckpoint:
    context: JobContext
    checkpoint_id: str
    step: int
    manifest_sha256: str
    files: tuple[tuple[str, StagedArtifact], ...]
    manifest: dict[str, Any]
    trainer_state: dict[str, Any]
    config: dict[str, Any]
    tokenizer_sha256: str
    config_sha256: str


@dataclass(frozen=True)
class CheckpointCommit:
    ack: dict[str, Any]
    checkpoint_id: str
    step: int
    sha256: str
    artifact_ids: tuple[str, ...]
    run_id: str


@dataclass(frozen=True)
class PreparedTerminal:
    context: JobContext | None
    state: str
    reason_code: str
    finished_at: str
    result: dict[str, Any] | None
    artifacts: tuple[PreparedArtifact, ...]
    discarded_artifacts: tuple[PreparedArtifact, ...]


@dataclass(frozen=True)
class TerminalCommit:
    result: dict[str, Any] | None
    run_id: str | None
    model_id: str | None
    artifact_ids: tuple[str, ...]


def _proposal_error(code: str, message: str, exc: BaseException | None = None) -> OperationStoreError:
    error = OperationStoreError(code, message)
    if exc is not None:
        error.__cause__ = exc
    return error


def _mapping_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = strict_json(raw)
    except (ApiError, TypeError, ValueError, UnicodeError) as exc:
        raise _proposal_error("WORKER_PROTOCOL_ERROR", f"{label} is not strict JSON.", exc)
    if not isinstance(value, Mapping):
        raise _proposal_error("WORKER_PROTOCOL_ERROR", f"{label} must be a JSON object.")
    return dict(value)


def _artifact_metadata(role: str, display_name: str) -> tuple[str, str, str, str]:
    json_roles = {
        "tokenizer_json",
        "evaluation_paired",
        "context_preview",
        "generation",
        "run_result",
    }
    jsonl_roles = {"training_metrics", "evaluation_metrics", "evaluation_records"}
    if role in json_roles:
        return role, display_name, "application/json", "text"
    if role in jsonl_roles:
        return role, display_name, "application/x-ndjson", "text"
    raise _proposal_error("WORKER_PROTOCOL_ERROR", "The artifact role is unsupported.")


def _checkpoint_artifact_metadata(checkpoint_id: str, name: str) -> tuple[str, str, str, str]:
    media_type = "application/json" if name.endswith(".json") else "application/octet-stream"
    preview = "metadata_only" if name.endswith(".json") else "download_only"
    return "checkpoint_file", f"{checkpoint_id}-{name}", media_type, preview


class OperationStore:
    """Own service-side copies until their checkpoint or terminal transaction commits."""

    def __init__(
        self,
        database: Database,
        registry: Registry,
        training_store: TrainingStore,
    ) -> None:
        self.database = database
        self.registry = registry
        self.training_store = training_store
        self._lock = threading.RLock()
        self._pending: dict[str, dict[str, PreparedArtifact]] = {}

    def _assert_output_capacity(
        self,
        job_id: str,
        *,
        proposed_bytes: int,
        proposed_rows: int,
        code: str | None,
        keep_terminal_capacity: bool,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        def check(current: sqlite3.Connection) -> None:
            reservation = current.execute(
                """
                SELECT byte_count, artifact_rows FROM reservations
                WHERE reservation_id = ? AND owner_kind = 'job'
                  AND owner_id = ? AND state = 'held'
                """,
                (job_id, job_id),
            ).fetchone()
            if reservation is None:
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
            committed = current.execute(
                """
                SELECT artifact_id, size_bytes FROM artifacts
                WHERE job_id = ? AND deleted_at IS NULL
                """,
                (job_id,),
            ).fetchall()
            committed_ids = {str(row[0]) for row in committed}
            committed_bytes = sum(int(row[1]) for row in committed)
            with self._lock:
                pending = tuple(self._pending.get(job_id, {}).values())
            pending = tuple(
                item for item in pending if item.artifact_id not in committed_ids
            )
            terminal_bytes = 16 * 1024 if keep_terminal_capacity else 0
            terminal_rows = 1 if keep_terminal_capacity else 0
            wanted_bytes = (
                committed_bytes
                + sum(item.staged.size for item in pending)
                + proposed_bytes
                + terminal_bytes
            )
            wanted_rows = (
                len(committed_ids) + len(pending) + proposed_rows + terminal_rows
            )
            if (
                wanted_bytes > int(reservation[0])
                or wanted_rows > int(reservation[1])
            ):
                message = "The proposed output exceeds the job's held reservation."
                if code is not None:
                    raise _proposal_error(code, message)
                raise ApiError(
                    "REGISTRY_LIMIT",
                    message,
                    reason_code="ROOT_QUOTA_EXCEEDED",
                )

        if connection is None:
            with self.database.read() as current:
                check(current)
        else:
            check(connection)
        try:
            free_bytes = shutil.disk_usage(self.registry.root).free
        except OSError as exc:
            if code is not None:
                raise _proposal_error(code, "Free storage could not be verified.", exc)
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT") from exc
        if free_bytes < proposed_bytes + FREE_SPACE_RESERVE_BYTES:
            message = "Free storage is below the reserve required for the service copy."
            if code is not None:
                raise _proposal_error(code, message)
            raise ApiError(
                "DISK_FULL", message, reason_code="INSUFFICIENT_STORAGE"
            )

    @staticmethod
    def _exact_proposal(value: Mapping[str, Any], keys: set[str], label: str) -> dict[str, Any]:
        if not isinstance(value, Mapping) or set(value) != keys:
            raise _proposal_error("WORKER_PROTOCOL_ERROR", f"{label} has invalid fields.")
        return dict(value)

    def _copy_staged_file(
        self,
        job_id: str,
        path: Path,
        *,
        expected_size: int,
        expected_sha256: str,
        code: str,
    ) -> StagedArtifact:
        try:
            _reject_linked_components(path, include_leaf=True)
            info = os.lstat(path)
            if is_reparse_point(path) or not stat.S_ISREG(info.st_mode):
                raise OSError("staged source is not a direct regular file")
            if info.st_size != expected_size:
                raise OSError("staged source size differs from its proposal")
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as source:
                opened = os.fstat(source.fileno())
                if not stat.S_ISREG(opened.st_mode) or opened.st_size != expected_size:
                    raise OSError("staged source changed before it was opened")
                staged = self.registry.stage_stream(
                    source,
                    job_id,
                    max_bytes=expected_size,
                    expected_sha256=expected_sha256,
                )
            if staged.size != expected_size or staged.sha256 != expected_sha256:
                staged.path.unlink(missing_ok=True)
                raise OSError("service copy differs from the proposal")
            return staged
        except ApiError as exc:
            raise _proposal_error(code, "The staged worker output failed verification.", exc)
        except (OSError, ValueError, TypeError) as exc:
            raise _proposal_error(code, "The staged worker output failed verification.", exc)

    def prepare_artifact(self, job_id: str, proposal: Mapping[str, Any]) -> PreparedArtifact:
        try:
            self.training_store.get_job_context(job_id)
        except ApiError as exc:
            raise _proposal_error("WORKER_PROTOCOL_ERROR", "The job has no S2 operation context.", exc)
        message = self._exact_proposal(
            proposal,
            {"type", "role", "staging_name", "size_bytes", "sha256"},
            "artifact proposal",
        )
        if message["type"] != "artifact_ready":
            raise _proposal_error("WORKER_PROTOCOL_ERROR", "The artifact proposal type is invalid.")
        role = message["role"]
        staging_name = message["staging_name"]
        size = message["size_bytes"]
        digest = message["sha256"]
        if (
            not isinstance(role, str)
            or not isinstance(staging_name, str)
            or _BASENAME.fullmatch(staging_name) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 <= size <= 1 << 40
            or not isinstance(digest, str)
            or _DIGEST.fullmatch(digest) is None
        ):
            raise _proposal_error("WORKER_PROTOCOL_ERROR", "The artifact proposal is invalid.")
        _artifact_metadata(role, staging_name)
        with self._lock:
            if role in self._pending.get(job_id, {}):
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "The artifact role was proposed twice.")
        self._assert_output_capacity(
            job_id,
            proposed_bytes=size,
            proposed_rows=1,
            code="WORKER_PROTOCOL_ERROR",
            keep_terminal_capacity=True,
        )
        source = self.registry.create_staging_dir(job_id) / staging_name
        staged = self._copy_staged_file(
            job_id,
            source,
            expected_size=size,
            expected_sha256=digest,
            code="WORKER_PROTOCOL_ERROR",
        )
        return PreparedArtifact(
            job_id=job_id,
            role=role,
            staging_name=staging_name,
            artifact_id=self.registry.allocate_artifact_id(),
            staged=staged,
        )

    def commit_artifact_in(
        self, connection: sqlite3.Connection, prepared: PreparedArtifact
    ) -> ArtifactCommit:
        del connection
        if not isinstance(prepared, PreparedArtifact):
            raise TypeError("prepared must be a PreparedArtifact")
        with self._lock:
            by_role = self._pending.setdefault(prepared.job_id, {})
            existing = by_role.get(prepared.role)
            if existing is not None and existing != prepared:
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "The artifact role was proposed twice.")
            by_role[prepared.role] = prepared
        return ArtifactCommit(
            ack={
                "type": "artifact_prepared",
                "role": prepared.role,
                "artifact_id": prepared.artifact_id,
                "sha256": prepared.staged.sha256,
            }
        )

    @staticmethod
    def _checkpoint_rows_in(
        connection: sqlite3.Connection, job_id: str
    ) -> list[dict[str, Any]]:
        rows = connection.execute(
            "SELECT checkpoint_id, record_json FROM checkpoints WHERE job_id = ? AND deleted_at IS NULL",
            (job_id,),
        )
        values: list[dict[str, Any]] = []
        for checkpoint_id, raw in rows:
            try:
                value = json.loads(str(raw))
                validate("checkpoint-descriptor.schema.json", value)
            except (TypeError, ValueError, json.JSONDecodeError, ApiError) as exc:
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT") from exc
            if not isinstance(value, dict) or value.get("checkpoint_id") != str(checkpoint_id):
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
            values.append(value)
        values.sort(key=lambda item: (int(item["step"]), str(item["checkpoint_id"])))
        return values

    def _next_checkpoint_id(
        self, connection: sqlite3.Connection, context: JobContext
    ) -> tuple[str, list[dict[str, Any]]]:
        committed = self._checkpoint_rows_in(connection, context.job_id)
        committed_ids = {str(item["checkpoint_id"]) for item in committed}
        prefix = set(context.checkpoint_ids[: len(committed)])
        if committed_ids != prefix or len(committed) >= len(context.checkpoint_ids):
            raise _proposal_error("WORKER_PROTOCOL_ERROR", "Checkpoint identity leases are inconsistent.")
        return context.checkpoint_ids[len(committed)], committed

    @staticmethod
    def _binding(value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "dataset_id": str(value["dataset_id"]),
            "split": str(value["split"]),
            "artifact_id": str(value["artifact_id"]),
            "sha256": str(value["sha256"]),
        }

    @staticmethod
    def _trainer_binding(value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "split": str(value["split"]),
            "dataset_id": str(value["dataset_id"]),
            "sha256": str(value["sha256"]),
        }

    def _validate_checkpoint_metadata(
        self,
        context: JobContext,
        checkpoint_id: str,
        staged: Mapping[str, StagedArtifact],
        manifest_sha256: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, str, int]:
        manifest_raw = staged["manifest.json"].path.read_bytes()
        trainer_raw = staged["trainer_state.json"].path.read_bytes()
        config_raw = staged["config.json"].path.read_bytes()
        tokenizer_raw = staged["tokenizer.json"].path.read_bytes()
        if hashlib.sha256(manifest_raw).hexdigest() != manifest_sha256:
            raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint manifest identity changed.")
        try:
            manifest = _mapping_bytes(manifest_raw, "manifest.json")
            trainer = _mapping_bytes(trainer_raw, "trainer_state.json")
            config = _mapping_bytes(config_raw, "config.json")
            validate("checkpoint-manifest.schema.json", manifest)
            validate("tiny-trainer-state.schema.json", trainer)
        except (ApiError, OperationStoreError) as exc:
            raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint JSON metadata is incompatible.", exc)
        if manifest.get("format") != "tiny-v2-checkpoint-v1" or manifest.get("portability") != "resume":
            raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint manifest is not Tiny-v2 resume format.")
        entries = manifest.get("files")
        if not isinstance(entries, list):
            raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint manifest file list is invalid.")
        manifest_files: dict[str, tuple[int, str]] = {}
        for raw in entries:
            if not isinstance(raw, Mapping):
                raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint manifest entry is invalid.")
            name = raw.get("name")
            size = raw.get("size_bytes")
            digest = raw.get("sha256")
            if not isinstance(name, str) or name in manifest_files:
                raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint manifest has duplicate files.")
            manifest_files[name] = (size, digest)  # type: ignore[assignment]
        if set(manifest_files) != _MANIFEST_PAYLOAD_NAMES:
            raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint manifest has the wrong file set.")
        for name in _CHECKPOINT_NAMES[:-1]:
            if manifest_files[name] != (staged[name].size, staged[name].sha256):
                raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint manifest does not bind its payload.")

        config_fields = {
            "format",
            "architecture_profile_id",
            "vocab_size",
            "context",
            "width",
            "heads",
            "layers",
        }
        if set(config) != config_fields or config.get("format") != "tiny-v2-config-v1":
            raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint config has invalid fields.")
        try:
            tokenizer = load_tokenizer_bytes(tokenizer_raw)
        except (TypeError, ValueError) as exc:
            raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint tokenizer is incompatible.", exc)
        if serialize_tokenizer(tokenizer) != tokenizer_raw:
            raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint tokenizer serialization is not exact.")
        tokenizer_sha256 = tokenizer.fingerprint()
        config_sha256 = hashlib.sha256(config_raw).hexdigest()
        if config.get("vocab_size") != tokenizer.vocab_size:
            raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint tokenizer and config differ.")

        expected_bindings = [
            self._trainer_binding(item)
            for item in context.resolved.get("dataset_bindings", [])
            if isinstance(item, Mapping)
        ]
        request = context.resolved.get("request")
        if not isinstance(request, Mapping):
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
        if context.operation == "tiny_train":
            expected_architecture = request.get("architecture_profile_id")
            expected_training = {
                "batch_size": request.get("batch_size"),
                "learning_rate": request.get("learning_rate"),
                "context": request.get("context"),
                "width": request.get("width"),
                "heads": request.get("heads"),
                "layers": request.get("layers"),
                "eval_every": request.get("eval_every"),
                "seed": request.get("seed"),
            }
            expected_tokenizer = context.resolved.get("tokenizer", {}).get("tokenizer_sha256")
        elif context.operation == "tiny_resume":
            parent = context.resolved.get("parent_checkpoint")
            training = context.resolved.get("training_identity")
            if not isinstance(parent, Mapping) or not isinstance(training, Mapping):
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
            model_identity = parent.get("model_identity")
            if not isinstance(model_identity, Mapping):
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
            expected_architecture = model_identity.get("architecture_profile_id")
            expected_tokenizer = model_identity.get("tokenizer_sha256")
            expected_training = {
                "batch_size": training.get("batch_size"),
                "learning_rate": training.get("learning_rate"),
                "context": training.get("context"),
                "width": training.get("width"),
                "heads": training.get("heads"),
                "layers": training.get("layers"),
                "eval_every": training.get("eval_every"),
                "seed": parent.get("seed"),
            }
        else:
            raise _proposal_error("WORKER_PROTOCOL_ERROR", "Only training jobs may propose checkpoints.")
        comparisons = {
            "architecture_profile_id": expected_architecture,
            "context": expected_training["context"],
            "width": expected_training["width"],
            "heads": expected_training["heads"],
            "layers": expected_training["layers"],
        }
        if any(config.get(key) != wanted for key, wanted in comparisons.items()):
            raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint config differs from admission.")
        trainer_comparisons = {
            "requested_final_step": context.resolved.get("requested_final_step"),
            "eval_every": expected_training["eval_every"],
            "batch_size": expected_training["batch_size"],
            "seed": expected_training["seed"],
            "learning_rate": expected_training["learning_rate"],
            "tokenizer_sha256": expected_tokenizer,
            "architecture_profile_id": expected_architecture,
            "runtime_profile": context.runtime_profile,
            "device": context.device,
            "dependency_lock_sha256": context.dependency_lock_sha256,
        }
        if any(trainer.get(key) != wanted for key, wanted in trainer_comparisons.items()):
            raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint trainer identity differs from admission.")
        if trainer.get("dataset_bindings") != expected_bindings:
            raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint dataset bindings differ from admission.")
        if tokenizer_sha256 != expected_tokenizer:
            raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint tokenizer fingerprint differs from admission.")
        step = trainer.get("completed_global_step")
        parameter_steps = trainer.get("parameter_steps")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or not isinstance(parameter_steps, Mapping)
            or any(value != step for value in parameter_steps.values())
        ):
            raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint optimizer steps are inconsistent.")
        incumbent = trainer.get("incumbent")
        if not isinstance(incumbent, Mapping):
            raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint incumbent is invalid.")
        incumbent_id = incumbent.get("checkpoint_id")
        if incumbent_id is not None:
            with self.database.read() as connection:
                same_job = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT checkpoint_id FROM checkpoints WHERE job_id = ? AND deleted_at IS NULL",
                        (context.job_id,),
                    )
                }
                durable = connection.execute(
                    "SELECT record_json FROM checkpoints WHERE checkpoint_id = ? AND deleted_at IS NULL",
                    (incumbent_id,),
                ).fetchone()
            allowed = same_job | {checkpoint_id}
            if context.operation == "tiny_resume":
                parent_id = str(request["checkpoint_id"])
                allowed.add(self._checkpoint_trainer_incumbent(parent_id))
            if incumbent_id not in allowed:
                raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint incumbent is outside this lineage.")
            if incumbent_id == checkpoint_id:
                incumbent_step = step
            else:
                if durable is None:
                    raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint incumbent is not durable or current.")
                try:
                    durable_value = json.loads(str(durable[0]))
                    incumbent_step = durable_value["step"]
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT") from exc
            if incumbent.get("step") != incumbent_step:
                raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint incumbent step is inconsistent.")
        return manifest, trainer, config, tokenizer_sha256, config_sha256, step

    def prepare_checkpoint(
        self, job_id: str, proposal: Mapping[str, Any]
    ) -> PreparedCheckpoint:
        context = self.training_store.get_job_context(job_id)
        if context.operation not in _TRAINING_OPERATIONS:
            raise _proposal_error("WORKER_PROTOCOL_ERROR", "This operation cannot commit checkpoints.")
        message = self._exact_proposal(
            proposal,
            {"type", "staging_name", "manifest_sha256", "files"},
            "checkpoint proposal",
        )
        if message["type"] != "checkpoint_ready":
            raise _proposal_error("WORKER_PROTOCOL_ERROR", "The checkpoint proposal type is invalid.")
        staging_name = message["staging_name"]
        digest = message["manifest_sha256"]
        raw_files = message["files"]
        if (
            not isinstance(staging_name, str)
            or _BASENAME.fullmatch(staging_name) is None
            or not isinstance(digest, str)
            or _DIGEST.fullmatch(digest) is None
            or not isinstance(raw_files, list)
            or len(raw_files) != len(_CHECKPOINT_NAMES)
        ):
            raise _proposal_error("WORKER_PROTOCOL_ERROR", "The checkpoint proposal is invalid.")
        proposed: dict[str, tuple[int, str]] = {}
        for raw in raw_files:
            item = self._exact_proposal(raw, {"name", "size", "sha256"}, "checkpoint file")
            name, size, item_digest = item["name"], item["size"], item["sha256"]
            if (
                not isinstance(name, str)
                or name in proposed
                or isinstance(size, bool)
                or not isinstance(size, int)
                or not 1 <= size <= 1 << 40
                or not isinstance(item_digest, str)
                or _DIGEST.fullmatch(item_digest) is None
            ):
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "The checkpoint file proposal is invalid.")
            proposed[name] = (size, item_digest)
        if set(proposed) != set(_CHECKPOINT_NAMES) or proposed["manifest.json"][1] != digest:
            raise _proposal_error("WORKER_PROTOCOL_ERROR", "The checkpoint file set is invalid.")
        self._assert_output_capacity(
            job_id,
            proposed_bytes=sum(size for size, _digest in proposed.values()),
            proposed_rows=len(proposed),
            code="CHECKPOINT_INCOMPATIBLE",
            keep_terminal_capacity=True,
        )

        directory = self.registry.create_staging_dir(job_id) / staging_name
        try:
            _reject_linked_components(directory, include_leaf=True)
            info = os.lstat(directory)
            if is_reparse_point(directory) or not stat.S_ISDIR(info.st_mode):
                raise OSError("checkpoint proposal is not a direct directory")
            entries = {entry.name: entry for entry in os.scandir(directory)}
            if set(entries) != set(_CHECKPOINT_NAMES):
                raise OSError("checkpoint directory has missing or extra entries")
            for entry in entries.values():
                entry_info = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or not stat.S_ISREG(entry_info.st_mode):
                    raise OSError("checkpoint directory contains a non-regular entry")
        except (OSError, ValueError) as exc:
            raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint staging is unsafe.", exc)

        staged: dict[str, StagedArtifact] = {}
        try:
            for name in _CHECKPOINT_NAMES:
                size, item_digest = proposed[name]
                staged[name] = self._copy_staged_file(
                    job_id,
                    directory / name,
                    expected_size=size,
                    expected_sha256=item_digest,
                    code="CHECKPOINT_INCOMPATIBLE",
                )
            with self.database.read() as connection:
                checkpoint_id, committed = self._next_checkpoint_id(connection, context)
            manifest, trainer, config, tokenizer_sha, config_sha, step = (
                self._validate_checkpoint_metadata(
                    context, checkpoint_id, staged, digest
                )
            )
            final_step = int(context.resolved["requested_final_step"])
            if step > final_step:
                raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint step exceeds the admitted final step.")
            if committed and step <= int(committed[-1]["step"]):
                raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint steps must increase within a run.")
            if not committed and context.operation == "tiny_train" and step != 0:
                raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Fresh training must first commit global step zero.")
            if not committed and context.operation == "tiny_resume":
                parent = context.resolved.get("parent_checkpoint")
                if not isinstance(parent, Mapping) or step <= int(parent["step"]):
                    raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Resume must not duplicate its inherited boundary.")
            return PreparedCheckpoint(
                context=context,
                checkpoint_id=checkpoint_id,
                step=step,
                manifest_sha256=digest,
                files=tuple((name, staged[name]) for name in _CHECKPOINT_NAMES),
                manifest=manifest,
                trainer_state=trainer,
                config=config,
                tokenizer_sha256=tokenizer_sha,
                config_sha256=config_sha,
            )
        except BaseException:
            for item in staged.values():
                item.path.unlink(missing_ok=True)
            raise

    def _run_manifest(self, context: JobContext, created_at: str) -> dict[str, Any]:
        request = context.resolved.get("request")
        if not isinstance(request, Mapping) or context.run_id is None:
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
        value: dict[str, Any] = {
            "format": "llm-foundations-run-v2",
            "run_id": context.run_id,
            "job_id": context.job_id,
            "operation": context.operation,
            "backend": "tiny_v2",
            "origin": "locally_created",
            "config": {},
            "dependency_lock_sha256": context.dependency_lock_sha256,
            "runtime_profile": context.runtime_profile,
            "device": context.device,
            "created_at": created_at,
        }
        bindings = context.resolved.get("dataset_bindings")
        if isinstance(bindings, list) and bindings:
            value["dataset_bindings"] = [
                self._binding(item) for item in bindings if isinstance(item, Mapping)
            ]
        if context.operation == "tokenizer_train":
            value["config"] = {"vocab_size": request["vocab_size"]}
            value["seed"] = request["seed"]
        elif context.operation == "tiny_train":
            value["config"] = {
                key: request[key]
                for key in (
                    "architecture_profile_id",
                    "steps",
                    "eval_every",
                    "batch_size",
                    "learning_rate",
                    "context",
                    "width",
                    "heads",
                    "layers",
                )
            }
            for key in ("exercise_profile_id", "curriculum_hold_after_step"):
                if key in request:
                    value["config"][key] = request[key]
            value["seed"] = request["seed"]
            value["model_id"] = context.model_id
            value["tokenizer_sha256"] = context.resolved["tokenizer"][
                "tokenizer_sha256"
            ]
        elif context.operation == "tiny_resume":
            training = context.resolved["training_identity"]
            parent = context.resolved["parent_checkpoint"]
            value["config"] = {
                "architecture_profile_id": parent["model_identity"]["architecture_profile_id"],
                "steps": request["additional_steps"],
                "eval_every": training["eval_every"],
                "batch_size": training["batch_size"],
                "learning_rate": training["learning_rate"],
                "context": training["context"],
                "width": training["width"],
                "heads": training["heads"],
                "layers": training["layers"],
            }
            value["seed"] = parent["seed"]
            value["parent_checkpoint_id"] = request["checkpoint_id"]
            value["model_id"] = context.model_id
            value["tokenizer_sha256"] = parent["model_identity"]["tokenizer_sha256"]
        elif context.operation == "evaluate":
            value["config"] = {
                "evaluation_profile_id": request["evaluation_profile_id"],
                "subject_ids": [item["checkpoint_id"] for item in request["subjects"]],
            }
        elif context.operation == "generate":
            checkpoint = context.resolved["checkpoint"]
            value["config"] = {
                key: request[key]
                for key in (
                    "preview_artifact_id",
                    "max_new_tokens",
                    "temperature",
                    "top_p",
                    "context_preview_digest",
                    "checkpoint_id",
                )
            }
            value["seed"] = request["seed"]
            value["model_id"] = checkpoint["model_id"]
            value["tokenizer_sha256"] = checkpoint["tokenizer_sha256"]
        else:
            raise ValueError("operation has no run manifest")
        try:
            validate("run-manifest.schema.json", value)
        except ApiError as exc:
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT") from exc
        return value

    def _ensure_run_in(
        self, connection: sqlite3.Connection, context: JobContext
    ) -> dict[str, Any] | None:
        if context.run_id is None:
            return None
        existing = connection.execute(
            "SELECT record_json FROM runs WHERE run_id = ?", (context.run_id,)
        ).fetchone()
        if existing is not None:
            try:
                value = json.loads(str(existing[0]))
                validate("run-manifest.schema.json", value)
            except (TypeError, ValueError, json.JSONDecodeError, ApiError) as exc:
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT") from exc
            if value.get("job_id") != context.job_id or value.get("operation") != context.operation:
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
            return value
        created_at = utc_now()
        value = self._run_manifest(context, created_at)
        connection.execute(
            """
            INSERT INTO runs(
                run_id, format, operation, origin, job_id, source_identity_json,
                created_at, updated_at, deleted_at, record_json
            ) VALUES (?, 'llm-foundations-run-v2', ?, 'locally_created', ?, NULL,
                      ?, ?, NULL, ?)
            """,
            (
                context.run_id,
                context.operation,
                context.job_id,
                created_at,
                created_at,
                canonical_json(value).decode("utf-8"),
            ),
        )
        return value

    def _checkpoint_descriptor(
        self,
        prepared: PreparedCheckpoint,
        *,
        artifact_ids: Mapping[str, str],
        previous: Mapping[str, Any] | None,
        created_at: str,
    ) -> dict[str, Any]:
        context = prepared.context
        request = context.resolved["request"]
        if context.operation == "tiny_train":
            training = request
            seed = request["seed"]
            parent_checkpoint_id = (
                None if previous is None else previous["checkpoint_id"]
            )
        else:
            training = context.resolved["training_identity"]
            parent = context.resolved["parent_checkpoint"]
            seed = parent["seed"]
            parent_checkpoint_id = (
                request["checkpoint_id"]
                if previous is None
                else previous["checkpoint_id"]
            )
        value = {
            "checkpoint_id": prepared.checkpoint_id,
            "artifact_id": artifact_ids["manifest.json"],
            "sha256": prepared.manifest_sha256,
            "format": "tiny_v2_portable",
            "origin": "locally_created",
            "run_id": context.run_id,
            "job_id": context.job_id,
            "backend": "tiny_v2",
            "step": prepared.step,
            "model_identity": {
                "kind": "tiny_v2",
                "architecture_profile_id": prepared.config["architecture_profile_id"],
                "config_sha256": prepared.config_sha256,
                "tokenizer_sha256": prepared.tokenizer_sha256,
            },
            "dataset_bindings": [
                self._binding(item) for item in context.resolved["dataset_bindings"]
            ],
            "training_identity": {
                "batch_size": training["batch_size"],
                "learning_rate": training["learning_rate"],
                "context": training["context"],
                "width": training["width"],
                "heads": training["heads"],
                "layers": training["layers"],
                "eval_every": training["eval_every"],
                "save_every": training["eval_every"],
            },
            "seed": seed,
            "dependency_lock_sha256": context.dependency_lock_sha256,
            "dtype": "float32",
            "runtime_profile": context.runtime_profile,
            "created_at": created_at,
            "portability": "resume",
            "contains_resume_state": True,
            "parent_checkpoint_id": parent_checkpoint_id,
            "device": context.device,
            "companion_source_revision": context.companion_source_revision,
        }
        try:
            validate("checkpoint-descriptor.schema.json", value)
        except ApiError as exc:
            raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint descriptor is incompatible.", exc)
        return value

    def commit_checkpoint_in(
        self, connection: sqlite3.Connection, prepared: PreparedCheckpoint
    ) -> CheckpointCommit:
        if not isinstance(prepared, PreparedCheckpoint):
            raise TypeError("prepared must be a PreparedCheckpoint")
        checkpoint_id, committed = self._next_checkpoint_id(connection, prepared.context)
        if checkpoint_id != prepared.checkpoint_id:
            raise _proposal_error("WORKER_PROTOCOL_ERROR", "The checkpoint lease changed before commit.")
        if committed and prepared.step <= int(committed[-1]["step"]):
            raise _proposal_error("CHECKPOINT_INCOMPATIBLE", "Checkpoint steps must increase within a run.")
        self._ensure_run_in(connection, prepared.context)
        artifact_ids: dict[str, str] = {}
        for name, staged in prepared.files:
            artifact_id = self.registry.allocate_artifact_id()
            artifact_type, display, media_type, preview = _checkpoint_artifact_metadata(
                prepared.checkpoint_id, name
            )
            descriptor = self.registry.commit_staged_in(
                connection,
                staged,
                artifact_type=artifact_type,
                display_name=display,
                media_type=media_type,
                preview_policy=preview,
                origin="locally_created",
                job_id=prepared.context.job_id,
                artifact_id=artifact_id,
                preserve_staged=True,
            )
            artifact_ids[name] = str(descriptor["artifact_id"])
        created_at = utc_now()
        previous = committed[-1] if committed else None
        checkpoint = self._checkpoint_descriptor(
            prepared,
            artifact_ids=artifact_ids,
            previous=previous,
            created_at=created_at,
        )
        connection.execute(
            """
            INSERT INTO checkpoints(
                checkpoint_id, format, backend, origin, run_id, job_id,
                manifest_sha256, source_identity_json, created_at, updated_at,
                deleted_at, record_json
            ) VALUES (?, 'tiny_v2_portable', 'tiny_v2', 'locally_created', ?, ?,
                      ?, NULL, ?, ?, NULL, ?)
            """,
            (
                prepared.checkpoint_id,
                prepared.context.run_id,
                prepared.context.job_id,
                prepared.manifest_sha256,
                created_at,
                created_at,
                canonical_json(checkpoint).decode("utf-8"),
            ),
        )
        for name, staged in prepared.files:
            connection.execute(
                """
                INSERT INTO checkpoint_files(
                    checkpoint_id, file_name, artifact_id, sha256, size_bytes
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    prepared.checkpoint_id,
                    name,
                    artifact_ids[name],
                    staged.sha256,
                    staged.size,
                ),
            )
        parent_id = checkpoint["parent_checkpoint_id"]
        if parent_id is not None:
            edge = (
                "resumed_from"
                if prepared.context.operation == "tiny_resume" and previous is None
                else "created_from"
            )
            connection.execute(
                """
                INSERT INTO lineage_edges(
                    source_type, source_id, target_type, target_id,
                    edge_type, creating_job_id, created_at
                ) VALUES ('checkpoint', ?, 'checkpoint', ?, ?, ?, ?)
                """,
                (parent_id, prepared.checkpoint_id, edge, prepared.context.job_id, created_at),
            )
        ack = {
            "type": "checkpoint_committed",
            "checkpoint_id": prepared.checkpoint_id,
            "step": prepared.step,
            "sha256": prepared.manifest_sha256,
            "run_id": prepared.context.run_id,
            "artifact_ids": artifact_ids,
        }
        return CheckpointCommit(
            ack=ack,
            checkpoint_id=prepared.checkpoint_id,
            step=prepared.step,
            sha256=prepared.manifest_sha256,
            artifact_ids=tuple(artifact_ids[name] for name in _CHECKPOINT_NAMES),
            run_id=str(prepared.context.run_id),
        )

    @staticmethod
    def _jsonl(path: Path, label: str) -> list[dict[str, Any]]:
        try:
            raw = path.read_bytes()
            if not raw or not raw.endswith(b"\n"):
                raise ValueError("JSONL must be nonempty and LF terminated")
            rows = []
            for line in raw[:-1].split(b"\n"):
                if not line:
                    raise ValueError("JSONL contains a blank line")
                value = strict_json(line)
                if not isinstance(value, Mapping):
                    raise ValueError("JSONL row is not an object")
                rows.append(dict(value))
            return rows
        except (ApiError, OSError, TypeError, ValueError, UnicodeError) as exc:
            raise _proposal_error("WORKER_PROTOCOL_ERROR", f"{label} is invalid JSONL.", exc)

    @staticmethod
    def _artifacts_by_role(artifacts: tuple[PreparedArtifact, ...]) -> dict[str, PreparedArtifact]:
        return {item.role: item for item in artifacts}

    def _validate_metric_artifact(
        self, artifact: PreparedArtifact, run_id: str
    ) -> list[dict[str, Any]]:
        rows = self._jsonl(artifact.staged.path, artifact.role)
        previous_sequence = -1
        previous_step = -1
        for row in rows:
            try:
                validate("metric-record.schema.json", row)
            except ApiError as exc:
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Metric output violates its schema.", exc)
            if row.get("run_id") != run_id:
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Metric output has the wrong run identity.")
            sequence = row.get("sequence")
            step = row.get("step")
            if (
                not isinstance(sequence, int)
                or isinstance(sequence, bool)
                or sequence <= previous_sequence
                or not isinstance(step, int)
                or isinstance(step, bool)
                or step < previous_step
            ):
                raise _proposal_error(
                    "WORKER_PROTOCOL_ERROR",
                    "Metric sequence must increase and metric step must not decrease.",
                )
            previous_sequence = sequence
            previous_step = step
        return rows

    def _validate_completed_result(
        self,
        context: JobContext,
        result: Mapping[str, Any] | None,
        artifacts: tuple[PreparedArtifact, ...],
    ) -> dict[str, Any]:
        if not isinstance(result, Mapping):
            raise _proposal_error("WORKER_PROTOCOL_ERROR", "A completed job requires a typed result.")
        try:
            normalized = validate("JobResult", dict(result))
        except ApiError as exc:
            raise _proposal_error("WORKER_PROTOCOL_ERROR", "The worker result violates JobResult.", exc)
        if not isinstance(normalized, dict) or normalized.get("operation") != context.operation:
            raise _proposal_error("WORKER_PROTOCOL_ERROR", "The worker result has the wrong operation.")
        by_role = self._artifacts_by_role(artifacts)
        artifact_ids = normalized.get("artifact_ids")
        if not isinstance(artifact_ids, list) or set(artifact_ids) != {
            item.artifact_id for item in artifacts
        }:
            raise _proposal_error("WORKER_PROTOCOL_ERROR", "The result artifact identities differ from prepared output.")
        request = context.resolved.get("request")
        if not isinstance(request, Mapping):
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")

        if context.operation == "tokenizer_train":
            if set(by_role) != {"tokenizer_json"} or normalized.get("tokenizer_id") != context.tokenizer_id:
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Tokenizer result identities are inconsistent.")
            raw = by_role["tokenizer_json"].staged.path.read_bytes()
            try:
                tokenizer = load_tokenizer_bytes(raw)
            except (TypeError, ValueError) as exc:
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Tokenizer output is invalid.", exc)
            if serialize_tokenizer(tokenizer) != raw or (
                normalized.get("tokenizer_sha256") != tokenizer.fingerprint()
                or normalized.get("vocab_size") != tokenizer.vocab_size
                or normalized.get("tokenizer_type") != tokenizer_type(tokenizer)
            ):
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Tokenizer result does not bind its bytes.")
        elif context.operation in _TRAINING_OPERATIONS:
            if set(by_role) != {"training_metrics"}:
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Training produced the wrong terminal artifacts.")
            if normalized.get("run_id") != context.run_id or normalized.get("model_id") != context.model_id:
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Training result identities are inconsistent.")
            self._validate_metric_artifact(by_role["training_metrics"], str(context.run_id))
            with self.database.read() as connection:
                checkpoints = self._checkpoint_rows_in(connection, context.job_id)
            if not checkpoints:
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Training completed without a durable checkpoint.")
            last = checkpoints[-1]
            if (
                normalized.get("last_checkpoint_id") != last["checkpoint_id"]
                or normalized.get("completed_step") != last["step"]
                or normalized.get("requested_final_step")
                != context.resolved.get("requested_final_step")
                or normalized.get("completed_step")
                != normalized.get("requested_final_step")
                or normalized.get("metrics_artifact_id")
                != by_role["training_metrics"].artifact_id
            ):
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Training result does not bind its durable boundary.")
            trainer_id = self._checkpoint_trainer_incumbent(str(last["checkpoint_id"]))
            if normalized.get("best_checkpoint_id") != trainer_id:
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Training result does not bind the checkpoint incumbent.")
            if context.operation == "tiny_resume" and normalized.get("parent_checkpoint_id") != request.get("checkpoint_id"):
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Resume result has the wrong parent checkpoint.")
        elif context.operation == "evaluate":
            wanted = {"evaluation_records", "evaluation_metrics"}
            if len(request.get("subjects", [])) == 2:
                wanted.add("evaluation_paired")
            if set(by_role) != wanted:
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Evaluation produced the wrong terminal artifacts.")
            if normalized.get("evaluation_id") != context.run_id or normalized.get("subjects") != request.get("subjects"):
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Evaluation result identities are inconsistent.")
            role_fields = {
                "metrics_artifact_id": "evaluation_metrics",
                "records_artifact_id": "evaluation_records",
            }
            for field, role in role_fields.items():
                if normalized.get(field) != by_role[role].artifact_id:
                    raise _proposal_error("WORKER_PROTOCOL_ERROR", "Evaluation result artifact binding is inconsistent.")
            metric_rows = self._validate_metric_artifact(
                by_role["evaluation_metrics"], str(context.run_id)
            )
            subjects = request.get("subjects")
            if not isinstance(subjects, list) or len(metric_rows) != len(subjects):
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Evaluation metric cardinality differs from its subjects.")
            bindings = context.resolved.get("dataset_bindings")
            if (
                not isinstance(bindings, list)
                or len(bindings) != 1
                or not isinstance(bindings[0], Mapping)
                or not isinstance(bindings[0].get("artifact_id"), str)
                or not isinstance(bindings[0].get("sealed"), bool)
            ):
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
            try:
                expected_records = expected_tiny_evaluation_records(
                    self.registry.read_artifact_bytes(
                        bindings[0]["artifact_id"],
                        allow_sealed_internal=bool(bindings[0]["sealed"]),
                    )
                )
                record_rows = self._jsonl(
                    by_role["evaluation_records"].staged.path,
                    "evaluation records",
                )
                validate_tiny_evaluation_records(
                    expected_records=expected_records,
                    record_rows=record_rows,
                    metric_rows=metric_rows,
                    subject_count=len(subjects),
                    records_artifact_id=by_role[
                        "evaluation_records"
                    ].artifact_id,
                )
            except (TypeError, ValueError, UnicodeError) as exc:
                raise _proposal_error(
                    "WORKER_PROTOCOL_ERROR",
                    "Evaluation records are invalid.",
                    exc,
                )
            for index, row in enumerate(metric_rows):
                payload = row.get("payload")
                if (
                    row.get("sequence") != index
                    or not isinstance(payload, Mapping)
                    or payload.get("records_artifact_id")
                    != by_role["evaluation_records"].artifact_id
                ):
                    raise _proposal_error("WORKER_PROTOCOL_ERROR", "Evaluation metrics do not bind their records artifact.")
            if "evaluation_paired" in by_role:
                if normalized.get("paired_artifact_id") != by_role["evaluation_paired"].artifact_id:
                    raise _proposal_error("WORKER_PROTOCOL_ERROR", "Paired evaluation identity is inconsistent.")
                self._validate_paired(context, by_role, metric_rows)
            elif normalized.get("paired_artifact_id") is not None:
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Single-subject evaluation cannot have a paired artifact.")
        elif context.operation == "context_preview":
            if set(by_role) != {"context_preview"} or normalized.get("preview_artifact_id") != by_role["context_preview"].artifact_id:
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Context preview artifact identity is inconsistent.")
            value = _mapping_bytes(by_role["context_preview"].staged.path.read_bytes(), "context preview")
            expected = {
                key: item
                for key, item in normalized.items()
                if key not in {"operation", "preview_artifact_id", "artifact_ids", "context_preview_digest"}
            }
            if value != expected or normalized.get("context_preview_digest") != hashlib.sha256(canonical_json(value)).hexdigest():
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Context preview result does not bind its artifact.")
        elif context.operation == "generate":
            if set(by_role) != {"generation"} or normalized.get("run_id") != context.run_id:
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Generation result identities are inconsistent.")
            if normalized.get("checkpoint_id") != request.get("checkpoint_id"):
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Generation result has the wrong checkpoint.")
            generated = _mapping_bytes(by_role["generation"].staged.path.read_bytes(), "generation")
            if (
                generated.get("job_id") != context.job_id
                or generated.get("run_id") != context.run_id
                or generated.get("text") != normalized.get("generated_text")
                or generated.get("generated_token_count") != normalized.get("generated_token_count")
                or generated.get("stop_reason") != normalized.get("stop_reason")
                or generated.get("serialized_input") != normalized.get("serialized_input")
            ):
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Generation result does not bind its artifact.")
        return normalized

    def _checkpoint_trainer_incumbent(self, checkpoint_id: str) -> str:
        with self.database.read() as connection:
            row = connection.execute(
                """
                SELECT a.artifact_id FROM checkpoint_files f
                JOIN artifacts a ON a.artifact_id = f.artifact_id
                WHERE f.checkpoint_id = ? AND f.file_name = 'trainer_state.json'
                  AND a.deleted_at IS NULL
                """,
                (checkpoint_id,),
            ).fetchone()
        if row is None:
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
        trainer = _mapping_bytes(
            self.registry.read_artifact_bytes(str(row[0])), "trainer_state.json"
        )
        incumbent = trainer.get("incumbent")
        if not isinstance(incumbent, Mapping) or not isinstance(incumbent.get("checkpoint_id"), str):
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
        return str(incumbent["checkpoint_id"])

    def _validate_paired(
        self,
        context: JobContext,
        by_role: Mapping[str, PreparedArtifact],
        metric_rows: list[dict[str, Any]],
    ) -> None:
        value = _mapping_bytes(
            by_role["evaluation_paired"].staged.path.read_bytes(),
            "paired evaluation",
        )
        try:
            validate_tiny_paired_evaluation(value)
        except (TypeError, ValueError) as exc:
            raise _proposal_error("WORKER_PROTOCOL_ERROR", "Paired evaluation is invalid.", exc)
        request = context.resolved["request"]
        binding = context.resolved["dataset_bindings"][0]
        expected = {
            "evaluation_id": context.run_id,
            "job_id": context.job_id,
            "dataset_id": request["dataset_id"],
            "dataset_manifest_sha256": context.resolved["dataset_manifest_sha256"],
            "evaluation_profile_id": request["evaluation_profile_id"],
            "records_artifact_id": by_role["evaluation_records"].artifact_id,
        }
        if any(value.get(key) != wanted for key, wanted in expected.items()) or binding["split"] != request["split"]:
            raise _proposal_error("WORKER_PROTOCOL_ERROR", "Paired evaluation identities differ from admission.")
        rows = self._jsonl(by_role["evaluation_records"].staged.path, "evaluation records")
        by_subject: dict[int, list[str]] = {0: [], 1: []}
        for row in rows:
            index = row.get("subject_index")
            record_id = row.get("record_id")
            if index not in by_subject or not isinstance(record_id, str):
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Evaluation records have invalid identities.")
            by_subject[index].append(record_id)
        ordered = value["ordered_record_ids"]
        if by_subject[0] != ordered or by_subject[1] != ordered:
            raise _proposal_error("WORKER_PROTOCOL_ERROR", "Paired evaluation record order differs from its records artifact.")
        subjects = context.resolved["subjects"]
        paired_subjects = value["subjects"]
        if len(metric_rows) != 2:
            raise _proposal_error("WORKER_PROTOCOL_ERROR", "Paired evaluation requires two metric rows.")
        for index in range(2):
            checkpoint = subjects[index]["checkpoint"]
            raw = paired_subjects[index]
            payload = metric_rows[index].get("payload")
            if not isinstance(payload, Mapping):
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Paired evaluation metric payload is missing.")
            checks = {
                "subject": request["subjects"][index],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "tokenizer_sha256": checkpoint["tokenizer_sha256"],
                "negative_log_likelihood_sum": payload["negative_log_likelihood_sum"],
                "utf8_bytes_sum": payload["utf8_bytes_sum"],
                "record_count": payload["record_count"],
                "nll_per_utf8_byte": payload["nll_per_utf8_byte"],
            }
            if any(raw.get(key) != wanted for key, wanted in checks.items()):
                raise _proposal_error("WORKER_PROTOCOL_ERROR", "Paired evaluation summary differs from metrics.")

    def prepare_terminal(
        self,
        job_id: str,
        *,
        state: str,
        reason_code: str,
        result: Mapping[str, Any] | None = None,
        finished_at: str | None = None,
    ) -> PreparedTerminal:
        if state not in _TERMINAL_STATES or not isinstance(reason_code, str):
            raise ValueError("terminal state or reason is invalid")
        terminal_time = finished_at or utc_now()
        try:
            context = self.training_store.get_job_context(job_id)
        except ApiError as exc:
            if exc.code != "NOT_FOUND":
                raise
            return PreparedTerminal(
                context=None,
                state=state,
                reason_code=reason_code,
                finished_at=terminal_time,
                result=None if result is None else dict(result),
                artifacts=(),
                discarded_artifacts=(),
            )
        with self._lock:
            artifacts = tuple(self._pending.get(job_id, {}).values())
        normalized = None
        retained = artifacts
        discarded: tuple[PreparedArtifact, ...] = ()
        if state == "completed":
            normalized = self._validate_completed_result(context, result, artifacts)
        elif result is not None:
            raise _proposal_error("WORKER_PROTOCOL_ERROR", "Failed or interrupted jobs cannot return a result.")
        else:
            retained_values: list[PreparedArtifact] = []
            if context.operation in _TRAINING_OPERATIONS and context.run_id is not None:
                metric = next(
                    (item for item in artifacts if item.role == "training_metrics"),
                    None,
                )
                if metric is not None:
                    try:
                        self._validate_metric_artifact(metric, context.run_id)
                    except OperationStoreError:
                        pass
                    else:
                        retained_values.append(metric)
            retained = tuple(retained_values)
            retained_set = set(retained)
            discarded = tuple(item for item in artifacts if item not in retained_set)
        return PreparedTerminal(
            context=context,
            state=state,
            reason_code=reason_code,
            finished_at=terminal_time,
            result=normalized,
            artifacts=retained,
            discarded_artifacts=discarded,
        )

    def _commit_terminal_artifacts_in(
        self,
        connection: sqlite3.Connection,
        prepared: PreparedTerminal,
    ) -> tuple[str, ...]:
        context = prepared.context
        assert context is not None
        artifact_ids: list[str] = []
        for artifact in prepared.artifacts:
            artifact_type, display, media_type, preview = _artifact_metadata(
                artifact.role, artifact.staging_name
            )
            descriptor = self.registry.commit_staged_in(
                connection,
                artifact.staged,
                artifact_type=artifact_type,
                display_name=display,
                media_type=media_type,
                preview_policy=preview,
                origin="locally_created",
                job_id=context.job_id,
                artifact_id=artifact.artifact_id,
                preserve_staged=True,
            )
            artifact_ids.append(str(descriptor["artifact_id"]))
            if context.run_id is not None:
                connection.execute(
                    "INSERT INTO run_artifacts(run_id, role, artifact_id) VALUES (?, ?, ?)",
                    (context.run_id, artifact.role, artifact.artifact_id),
                )
        return tuple(artifact_ids)

    def _register_tokenizer_in(
        self,
        connection: sqlite3.Connection,
        context: JobContext,
        result: Mapping[str, Any],
    ) -> None:
        artifact_id = str(result["artifact_ids"][0])
        created_at = utc_now()
        record = {
            "tokenizer_id": context.tokenizer_id,
            "tokenizer_type": result["tokenizer_type"],
            "vocab_size": result["vocab_size"],
            "tokenizer_sha256": result["tokenizer_sha256"],
            "created_at": created_at,
            "artifact_ids": [artifact_id],
        }
        validate("TokenizerDetail", record)
        connection.execute(
            """
            INSERT INTO tokenizers(
                tokenizer_id, tokenizer_type, vocab_size, tokenizer_sha256,
                artifact_id, run_id, job_id, origin, created_at, updated_at,
                deleted_at, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'locally_created', ?, ?, NULL, ?)
            """,
            (
                context.tokenizer_id,
                result["tokenizer_type"],
                result["vocab_size"],
                result["tokenizer_sha256"],
                artifact_id,
                context.run_id,
                context.job_id,
                created_at,
                created_at,
                canonical_json(record).decode("utf-8"),
            ),
        )

    def _register_model_if_checkpoint_in(
        self,
        connection: sqlite3.Connection,
        context: JobContext,
    ) -> str | None:
        if context.operation not in _TRAINING_OPERATIONS or context.model_id is None:
            return None
        checkpoints = self._checkpoint_rows_in(connection, context.job_id)
        if not checkpoints:
            return None
        checkpoint = checkpoints[-1]
        existing = connection.execute(
            "SELECT record_json FROM models WHERE model_id = ?", (context.model_id,)
        ).fetchone()
        if existing is not None:
            try:
                record = json.loads(str(existing[0]))
                validate("ModelDetail", record)
            except (TypeError, ValueError, json.JSONDecodeError, ApiError) as exc:
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT") from exc
            if record.get("checkpoint_id") != checkpoint["checkpoint_id"]:
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
            return context.model_id
        rows = connection.execute(
            "SELECT artifact_id FROM checkpoint_files WHERE checkpoint_id = ? ORDER BY file_name",
            (checkpoint["checkpoint_id"],),
        )
        artifact_ids = [str(row[0]) for row in rows]
        created_at = utc_now()
        record = {
            "model_id": context.model_id,
            "kind": "tiny",
            "display_name": f"TinyLM {context.model_id[:8]}",
            "checkpoint_id": checkpoint["checkpoint_id"],
            "checkpoint_sha256": checkpoint["sha256"],
            "created_at": created_at,
            "artifact_ids": artifact_ids,
        }
        validate("ModelDetail", record)
        connection.execute(
            """
            INSERT INTO models(
                model_id, format, backend, origin, creating_job_id,
                source_identity_json, created_at, updated_at, deleted_at,
                record_json
            ) VALUES (?, 'llm-foundations-model-v1', 'tiny_v2', 'locally_created', ?,
                      NULL, ?, ?, NULL, ?)
            """,
            (
                context.model_id,
                context.job_id,
                created_at,
                created_at,
                canonical_json(record).decode("utf-8"),
            ),
        )
        return context.model_id

    def _commit_run_result_in(
        self,
        connection: sqlite3.Connection,
        prepared: PreparedTerminal,
    ) -> str | None:
        context = prepared.context
        assert context is not None
        if context.run_id is None:
            return None
        row = connection.execute(
            "SELECT record_json FROM jobs WHERE job_id = ?", (context.job_id,)
        ).fetchone()
        if row is None:
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
        try:
            decoded = strict_json(str(row[0]).encode("utf-8"))
        except (TypeError, ValueError, UnicodeError) as exc:
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT") from exc
        if not isinstance(decoded, Mapping):
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
        job = decoded
        if job.get("run_id") != context.run_id:
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
        boundary = job.get("checkpoint_boundary")
        if boundary is None:
            durable_step: int | None = None
        elif (
            isinstance(boundary, Mapping)
            and isinstance(boundary.get("step"), int)
            and not isinstance(boundary.get("step"), bool)
        ):
            durable_step = int(boundary["step"])
        else:
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
        raw_observed = job.get("step")
        if not isinstance(raw_observed, int) or isinstance(raw_observed, bool):
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
        observed_step = raw_observed
        projection: dict[str, Any] = {
            "run_id": context.run_id,
            "status": prepared.state,
            "last_observed_step": observed_step,
            "last_durable_checkpoint_step": durable_step,
            "finished_at": prepared.finished_at,
        }
        requested = job.get("requested_final_step")
        if context.operation not in _TRAINING_OPERATIONS:
            if requested is not None:
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
            requested = 0
        if prepared.state == "completed":
            if not isinstance(requested, int) or isinstance(requested, bool):
                raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
            projection["requested_final_step"] = requested
            projection["final_step"] = observed_step
        elif isinstance(requested, int) and not isinstance(requested, bool):
            projection["requested_final_step"] = requested
        elif requested is not None:
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT")
        if (
            context.operation in _TRAINING_OPERATIONS
            and prepared.state == "completed"
            and observed_step != requested
        ):
            raise _proposal_error(
                "WORKER_PROTOCOL_ERROR",
                "Completed training did not reach its requested final step.",
            )
        if prepared.state == "failed":
            projection["error"] = prepared.reason_code[:4000]
        try:
            validate("run-result.schema.json", projection)
        except ApiError as exc:
            raise ApiError("STORAGE_UNAVAILABLE", reason_code="STORAGE_CORRUPT") from exc
        raw = canonical_json(projection)
        self._assert_output_capacity(
            context.job_id,
            proposed_bytes=len(raw),
            proposed_rows=1,
            code=None,
            keep_terminal_capacity=False,
            connection=connection,
        )
        staged = self.registry.stage_stream(
            raw,
            context.job_id,
            max_bytes=16 * 1024,
            expected_sha256=hashlib.sha256(raw).hexdigest(),
        )
        artifact_id = self.registry.allocate_artifact_id()
        descriptor = self.registry.commit_staged_in(
            connection,
            staged,
            artifact_type="run_result",
            display_name="run-result.json",
            media_type="application/json",
            preview_policy="text",
            origin="locally_created",
            job_id=context.job_id,
            artifact_id=artifact_id,
        )
        connection.execute(
            "INSERT INTO run_artifacts(run_id, role, artifact_id) VALUES (?, 'run_result', ?)",
            (context.run_id, artifact_id),
        )
        return str(descriptor["artifact_id"])

    def finalize_terminal_in(
        self,
        connection: sqlite3.Connection,
        prepared: PreparedTerminal,
    ) -> TerminalCommit:
        if not isinstance(prepared, PreparedTerminal):
            raise TypeError("prepared must be a PreparedTerminal")
        context = prepared.context
        if context is None:
            return TerminalCommit(
                result=prepared.result,
                run_id=None,
                model_id=None,
                artifact_ids=(),
            )
        self._ensure_run_in(connection, context)
        artifact_ids = list(self._commit_terminal_artifacts_in(connection, prepared))
        if prepared.state == "completed" and prepared.result is not None:
            if context.operation == "tokenizer_train":
                self._register_tokenizer_in(connection, context, prepared.result)
        model_id = self._register_model_if_checkpoint_in(connection, context)
        run_result_id = self._commit_run_result_in(connection, prepared)
        if run_result_id is not None:
            artifact_ids.append(run_result_id)
        return TerminalCommit(
            result=prepared.result,
            run_id=context.run_id,
            model_id=model_id,
            artifact_ids=tuple(artifact_ids),
        )

    def complete_terminal(self, prepared: PreparedTerminal) -> None:
        """Release exact pending service copies after the caller commits."""

        if not isinstance(prepared, PreparedTerminal):
            raise TypeError("prepared must be a PreparedTerminal")
        context = prepared.context
        if context is None:
            return
        items = prepared.artifacts + prepared.discarded_artifacts
        with self._lock:
            current = self._pending.get(context.job_id, {})
            for item in items:
                if current.get(item.role) == item:
                    current.pop(item.role, None)
                    try:
                        item.staged.path.unlink(missing_ok=True)
                    except OSError:
                        pass
            if not current:
                self._pending.pop(context.job_id, None)

    @staticmethod
    def complete_checkpoint(prepared: PreparedCheckpoint) -> None:
        """Release checkpoint service copies after the caller commits."""

        if not isinstance(prepared, PreparedCheckpoint):
            raise TypeError("prepared must be a PreparedCheckpoint")
        for _name, staged in prepared.files:
            try:
                staged.path.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = [
    "ArtifactCommit",
    "CheckpointCommit",
    "OperationStore",
    "OperationStoreError",
    "PreparedArtifact",
    "PreparedCheckpoint",
    "PreparedTerminal",
    "TerminalCommit",
]
