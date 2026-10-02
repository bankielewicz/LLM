"""Worker-only access to verified inputs and scheduler-owned output commits."""

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO

from .worker_protocol import (
    ARTIFACT_ROLES,
    CancellationToken,
    WorkerProtocolError,
    read_frame,
    validate_event_payload,
    validate_input_snapshot,
    validate_output_allocations,
    validate_parent_message,
    validate_worker_message,
    write_frame,
)


@dataclass(frozen=True)
class VerifiedInput:
    role: str
    artifact_id: str
    sha256: str
    size_bytes: int
    path: Path
    descriptor: Mapping[str, Any]


class CommitRejected(RuntimeError):
    """The parent rejected a staged artifact or checkpoint proposal."""

    def __init__(self, kind: str, error: Mapping[str, Any]) -> None:
        super().__init__(str(error.get("message", "The staged output was rejected.")))
        self.kind = kind
        self.error = dict(error)


class WorkerInterrupted(RuntimeError):
    """A handler reached a specified interrupted terminal boundary."""

    def __init__(
        self, reason_code: str, error: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.error = dict(error) if error is not None else None


def _portable_basename(value: str) -> str:
    candidate = Path(value)
    if (
        not value
        or len(value) > 120
        or candidate.name != value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise WorkerProtocolError("staging filename must be a portable basename")
    return value


def _digest_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _fsync_file(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class WorkerContext:
    """Capability object given to one fixed operation handler."""

    def __init__(
        self,
        *,
        job_id: str,
        instance_id: str,
        runtime_profile: str,
        staging_path: Path,
        cancellation: CancellationToken,
        protocol: BinaryIO,
        control: BinaryIO,
        input_snapshot: Mapping[str, Any],
        output_allocations: Mapping[str, Any],
    ) -> None:
        self.job_id = job_id
        self.instance_id = instance_id
        self.runtime_profile = runtime_profile
        self.staging_path = staging_path.resolve()
        self.cancellation = cancellation
        self._protocol = protocol
        self._control = control
        snapshot = validate_input_snapshot(input_snapshot)
        allocations = validate_output_allocations(output_allocations)
        if dict(snapshot["ids"]) != dict(allocations):
            raise WorkerProtocolError("worker allocations do not match the verified snapshot")
        if snapshot["job_id"] != job_id or snapshot["runtime_profile"] != runtime_profile:
            raise WorkerProtocolError("worker context identity does not match the snapshot")
        self.snapshot = MappingProxyType(dict(snapshot))
        self.output_allocations = MappingProxyType(dict(allocations))
        self._inputs = {str(item["role"]): dict(item) for item in snapshot["inputs"]}
        self._verified_inputs: dict[str, VerifiedInput] = {}
        self._checkpoint_index = 0
        self._last_checkpoint: Mapping[str, Any] | None = None
        if snapshot["operation"] == "tiny_resume":
            parent = snapshot["resolved"].get("parent_checkpoint")
            if not isinstance(parent, Mapping):
                raise WorkerProtocolError("resume snapshot has no verified parent checkpoint")
            checkpoint_id = parent.get("checkpoint_id")
            step = parent.get("step")
            try:
                parsed_id = uuid.UUID(str(checkpoint_id))
            except (ValueError, TypeError, AttributeError) as exc:
                raise WorkerProtocolError(
                    "resume parent checkpoint identity is invalid"
                ) from exc
            if str(parsed_id) != checkpoint_id:
                raise WorkerProtocolError("resume parent checkpoint identity is invalid")
            if (
                isinstance(step, bool)
                or not isinstance(step, int)
                or not 0 <= step <= 2_147_483_647
            ):
                raise WorkerProtocolError("resume parent checkpoint step is invalid")
            self._last_checkpoint = MappingProxyType(
                {"checkpoint_id": checkpoint_id, "step": step}
            )

    @property
    def resolved(self) -> Mapping[str, Any]:
        return self.snapshot["resolved"]

    @property
    def last_checkpoint(self) -> Mapping[str, Any] | None:
        return self._last_checkpoint

    def emit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        validate_event_payload(event_type, payload)
        if event_type in {"state_changed", "checkpoint_committed", "terminal"}:
            raise WorkerProtocolError("the scheduler owns durable state events")
        write_frame(
            self._protocol,
            {"type": "event", "event_type": event_type, "payload": dict(payload)},
        )

    def input(self, role: str) -> VerifiedInput:
        if role in self._verified_inputs:
            return self._verified_inputs[role]
        try:
            descriptor = self._inputs[role]
        except KeyError as exc:
            raise WorkerProtocolError(f"verified input role is unavailable: {role}") from exc
        relative = Path(str(descriptor["path"]))
        if relative.parts[:1] != ("inputs",) or len(relative.parts) != 2:
            raise WorkerProtocolError("verified input path is not confined")
        candidate = self.staging_path / relative
        candidate_info = os.lstat(candidate)
        if stat.S_ISLNK(candidate_info.st_mode):
            raise WorkerProtocolError("verified input path is a link")
        path = candidate.resolve(strict=True)
        input_root = (self.staging_path / "inputs").resolve(strict=True)
        if path.parent != input_root:
            raise WorkerProtocolError("verified input path escaped its fixed directory")
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode):
            raise WorkerProtocolError("verified input is not a regular file")
        size, digest = _digest_file(path)
        if size != descriptor["size_bytes"] or digest != descriptor["sha256"]:
            raise WorkerProtocolError("verified input bytes changed before worker use")
        value = VerifiedInput(
            role=role,
            artifact_id=str(descriptor["artifact_id"]),
            sha256=digest,
            size_bytes=size,
            path=path,
            descriptor=MappingProxyType(dict(descriptor)),
        )
        self._verified_inputs[role] = value
        return value

    def checkpoint_identity(self) -> str:
        checkpoint_ids = self.output_allocations["checkpoint_ids"]
        if self._checkpoint_index >= len(checkpoint_ids):
            raise WorkerProtocolError("worker exhausted its preallocated checkpoint identities")
        return str(checkpoint_ids[self._checkpoint_index])

    def _read_ack(self, expected_kind: str) -> Mapping[str, Any]:
        message = validate_parent_message(read_frame(self._control))
        if message["type"] == "commit_rejected":
            if message["kind"] != expected_kind:
                raise WorkerProtocolError("commit rejection kind does not match proposal")
            raise CommitRejected(expected_kind, message["error"])
        actual = "artifact" if message["type"] == "artifact_prepared" else "checkpoint"
        if actual != expected_kind:
            raise WorkerProtocolError("commit acknowledgment kind does not match proposal")
        return message

    def stage_artifact(
        self,
        role: str,
        data: bytes | bytearray | memoryview | Path,
        *,
        filename: str | None = None,
    ) -> Mapping[str, Any]:
        if role not in ARTIFACT_ROLES:
            raise WorkerProtocolError("artifact role is outside the closed vocabulary")
        if isinstance(data, Path):
            source_info = os.lstat(data)
            if stat.S_ISLNK(source_info.st_mode):
                raise WorkerProtocolError("staged artifact is a link")
            path = data.resolve(strict=True)
            if path.parent != self.staging_path:
                raise WorkerProtocolError("staged artifact is outside the worker staging root")
        else:
            name = _portable_basename(filename or f"{uuid.uuid4()}.partial")
            path = self.staging_path / name
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags, 0o600)
            with os.fdopen(descriptor, "wb", buffering=0) as stream:
                view = memoryview(bytes(data))
                while view:
                    written = stream.write(view)
                    if written is None or written <= 0:
                        raise OSError("short staged artifact write")
                    view = view[written:]
                stream.flush()
                os.fsync(stream.fileno())
            _fsync_directory(self.staging_path)
        _fsync_file(path)
        _fsync_directory(self.staging_path)
        size, digest = _digest_file(path)
        proposal = {
            "type": "artifact_ready",
            "role": role,
            "staging_name": _portable_basename(path.name),
            "size_bytes": size,
            "sha256": digest,
        }
        validate_worker_message(proposal)
        write_frame(self._protocol, proposal)
        acknowledgment = self._read_ack("artifact")
        if acknowledgment["role"] != role or acknowledgment["sha256"] != digest:
            raise WorkerProtocolError("artifact acknowledgment does not match proposal")
        path.unlink()
        _fsync_directory(self.staging_path)
        return acknowledgment

    def commit_checkpoint(self, proposal: Mapping[str, Any]) -> Mapping[str, Any]:
        message = {"type": "checkpoint_ready", **dict(proposal)}
        validate_worker_message(message)
        directory_name = _portable_basename(str(message["staging_name"]))
        candidate = self.staging_path / directory_name
        info = os.lstat(candidate)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise WorkerProtocolError("checkpoint staging entry is not a directory")
        directory = candidate.resolve(strict=True)
        if directory.parent != self.staging_path:
            raise WorkerProtocolError("checkpoint staging directory escaped its job root")
        declared_names = {str(item["name"]) for item in message["files"]}
        actual_names = {item.name for item in directory.iterdir()}
        if actual_names != declared_names:
            raise WorkerProtocolError("checkpoint staging directory has undeclared files")
        for item in message["files"]:
            path = directory / str(item["name"])
            file_info = os.lstat(path)
            if stat.S_ISLNK(file_info.st_mode) or not stat.S_ISREG(file_info.st_mode):
                raise WorkerProtocolError("checkpoint member is not a regular file")
            size, digest = _digest_file(path)
            if size != item["size"] or digest != item["sha256"]:
                raise WorkerProtocolError("checkpoint member bytes do not match the proposal")
            _fsync_file(path)
        _fsync_directory(directory)
        _fsync_directory(self.staging_path)
        expected_id = self.checkpoint_identity()
        write_frame(self._protocol, message)
        try:
            acknowledgment = self._read_ack("checkpoint")
        except CommitRejected as exc:
            if self.cancellation.cancelled and exc.error.get("code") == "CHECKPOINT_WRITE_FAILED":
                self.interrupt(self.cancellation.reason or "user_cancelled", exc.error)
            raise
        if acknowledgment["checkpoint_id"] != expected_id:
            raise WorkerProtocolError("checkpoint acknowledgment used an unleased identity")
        if acknowledgment["run_id"] != self.output_allocations["run_id"]:
            raise WorkerProtocolError("checkpoint acknowledgment used an unleased run identity")
        if acknowledgment["sha256"] != message["manifest_sha256"]:
            raise WorkerProtocolError("checkpoint acknowledgment digest does not match proposal")
        for item in message["files"]:
            (directory / str(item["name"])).unlink()
        _fsync_directory(directory)
        directory.rmdir()
        _fsync_directory(self.staging_path)
        self._checkpoint_index += 1
        self._last_checkpoint = MappingProxyType(dict(acknowledgment))
        return acknowledgment

    def interrupt(
        self, reason_code: str, error: Mapping[str, Any] | None = None
    ) -> None:
        raise WorkerInterrupted(reason_code, error)


__all__ = [
    "CommitRejected",
    "VerifiedInput",
    "WorkerContext",
    "WorkerInterrupted",
]
