"""Framed parent/worker protocol with closed semantic validation.

The request, cancellation, and protocol channels are inherited anonymous pipes.
Frames are four-byte big-endian lengths followed by canonical UTF-8 JSON.  The
worker never reads request data from argv, environment values, or stdin.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import struct
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, BinaryIO

from .errors import ERROR_VOCABULARY
from .operations import OPERATIONS
from .schema import canonical_json, strict_json


PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 1_048_576
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_BASENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
ARTIFACT_ROLES = frozenset(
    {
        "tokenizer_json",
        "training_metrics",
        "evaluation_metrics",
        "evaluation_records",
        "evaluation_paired",
        "context_preview",
        "generation",
        "run_result",
    }
)
_EVENT_TYPES = frozenset(
    {
        "state_changed",
        "phase_changed",
        "progress",
        "metric",
        "warning",
        "checkpoint_committed",
        "terminal",
    }
)
_PHASES = frozenset(
    {
        "loading",
        "initial_evaluation",
        "training",
        "evaluation",
        "cancellable_hold",
        "checkpointing",
        "generating",
        "downloading",
        "validating",
        "importing",
        "exporting",
    }
)
_STATES = frozenset(
    {"queued", "starting", "running", "cancelling", "completed", "failed", "interrupted"}
)
_TERMINAL_STATES = frozenset({"completed", "failed", "interrupted"})
_TERMINAL_REASONS = frozenset(
    {
        "completed",
        "exercise_timeout",
        "service_restarted",
        "service_shutdown",
        "timeout",
        "user_cancelled",
        "worker_lost",
        "worker_ownership_unknown",
    }
)
_PROGRESS_UNITS = frozenset({"updates", "records", "tokens", "bytes", "files"})
_JOB_ERROR_CODES = frozenset(
    code
    for code, row in ERROR_VOCABULARY.items()
    if "job" in row.get("kinds", ())
)


class WorkerProtocolError(ValueError):
    """Malformed, inconsistent, or out-of-order worker protocol data."""


def _canonical_uuid(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise WorkerProtocolError(f"{field} must be a lowercase UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, TypeError, AttributeError) as exc:
        raise WorkerProtocolError(f"{field} must be a lowercase UUID") from exc
    if str(parsed) != value:
        raise WorkerProtocolError(f"{field} must be a lowercase UUID")
    return value


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise WorkerProtocolError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _exact(value: object, keys: set[str], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise WorkerProtocolError(f"{field} has invalid fields")
    return value


def _bounded_int(value: object, field: str, *, maximum: int = 2_147_483_647) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise WorkerProtocolError(f"{field} is outside its integer bounds")
    return value


def _bounded_text(value: object, field: str, minimum: int, maximum: int) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        raise WorkerProtocolError(f"{field} is outside its text bounds")
    return value


def _basename(value: object, field: str) -> str:
    if not isinstance(value, str) or _BASENAME.fullmatch(value) is None:
        raise WorkerProtocolError(f"{field} must be a portable basename")
    return value


def _nullable_uuid(value: object, field: str) -> str | None:
    return None if value is None else _canonical_uuid(value, field)


def validate_output_allocations(value: object) -> Mapping[str, Any]:
    allocations = _exact(
        value,
        {"run_id", "model_id", "tokenizer_id", "checkpoint_ids"},
        "output allocations",
    )
    for name in ("run_id", "model_id", "tokenizer_id"):
        _nullable_uuid(allocations[name], name)
    checkpoint_ids = allocations["checkpoint_ids"]
    if not isinstance(checkpoint_ids, list) or len(checkpoint_ids) > 2_002:
        raise WorkerProtocolError("checkpoint_ids is invalid")
    seen: set[str] = set()
    for index, checkpoint_id in enumerate(checkpoint_ids):
        parsed = _canonical_uuid(checkpoint_id, f"checkpoint_ids[{index}]")
        if parsed in seen:
            raise WorkerProtocolError("checkpoint_ids contains a duplicate")
        seen.add(parsed)
    return allocations


def validate_input_snapshot(value: object) -> Mapping[str, Any]:
    snapshot = _exact(
        value,
        {
            "format",
            "job_id",
            "operation",
            "request_sha256",
            "runtime_profile",
            "device",
            "dependency_lock_sha256",
            "companion_source_revision",
            "ids",
            "resolved",
            "inputs",
        },
        "input snapshot",
    )
    if snapshot["format"] != "llm-foundations-worker-input-v1":
        raise WorkerProtocolError("input snapshot format is invalid")
    _canonical_uuid(snapshot["job_id"], "input snapshot job_id")
    if snapshot["operation"] not in OPERATIONS:
        raise WorkerProtocolError("input snapshot operation is invalid")
    _digest(snapshot["request_sha256"], "input snapshot request_sha256")
    if snapshot["runtime_profile"] not in {"win-cpu", "win-cuda", "wsl-cpu", "wsl-cuda"}:
        raise WorkerProtocolError("input snapshot runtime profile is invalid")
    if snapshot["device"] not in {"cpu", "cuda"}:
        raise WorkerProtocolError("input snapshot device is invalid")
    _digest(snapshot["dependency_lock_sha256"], "dependency_lock_sha256")
    revision = snapshot["companion_source_revision"]
    if not isinstance(revision, str) or re.fullmatch(r"^[0-9a-f]{40}$", revision) is None:
        raise WorkerProtocolError("companion_source_revision is invalid")
    validate_output_allocations(snapshot["ids"])
    if not isinstance(snapshot["resolved"], Mapping):
        raise WorkerProtocolError("input snapshot resolved value must be an object")
    inputs = snapshot["inputs"]
    if not isinstance(inputs, list) or len(inputs) > 100:
        raise WorkerProtocolError("input snapshot inputs is invalid")
    roles: set[str] = set()
    for index, raw in enumerate(inputs):
        item = _exact(
            raw,
            {"role", "artifact_id", "sha256", "size_bytes", "path"},
            f"input snapshot inputs[{index}]",
        )
        role = _bounded_text(item["role"], "input role", 1, 120)
        if role in roles:
            raise WorkerProtocolError("input snapshot roles must be unique")
        roles.add(role)
        _canonical_uuid(item["artifact_id"], "input artifact_id")
        _digest(item["sha256"], "input sha256")
        _bounded_int(item["size_bytes"], "input size_bytes", maximum=1 << 40)
        path = _bounded_text(item["path"], "input path", 1, 240)
        if re.fullmatch(r"inputs/[A-Za-z0-9][A-Za-z0-9._-]{0,119}", path) is None:
            raise WorkerProtocolError("input path must be a fixed relative staging path")
    return snapshot


def encode_frame(value: Mapping[str, Any]) -> bytes:
    body = canonical_json(dict(value))
    if len(body) > MAX_FRAME_BYTES:
        raise WorkerProtocolError("protocol frame exceeds one MiB")
    return struct.pack(">I", len(body)) + body


def write_frame(stream: BinaryIO, value: Mapping[str, Any]) -> None:
    stream.write(encode_frame(value))
    stream.flush()


def _read_exact(stream: BinaryIO, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        part = stream.read(remaining)
        if not part:
            raise EOFError("protocol pipe closed inside a frame")
        chunks.append(part)
        remaining -= len(part)
    return b"".join(chunks)


def read_frame(stream: BinaryIO, *, allow_eof: bool = False) -> Mapping[str, Any] | None:
    header = stream.read(4)
    if not header:
        if allow_eof:
            return None
        raise EOFError("protocol pipe closed before a frame")
    if len(header) != 4:
        raise EOFError("protocol pipe closed inside a frame header")
    length = struct.unpack(">I", header)[0]
    if length == 0 or length > MAX_FRAME_BYTES:
        raise WorkerProtocolError("protocol frame length is invalid")
    try:
        value = strict_json(_read_exact(stream, length))
    except Exception as exc:
        raise WorkerProtocolError("protocol frame is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise WorkerProtocolError("protocol frame must be a JSON object")
    return value


def make_request_envelope(
    *,
    job_id: str,
    instance_id: str,
    spawn_nonce: str,
    schema_id: str,
    request_sha256: str,
    request: Mapping[str, Any],
    input_snapshot_sha256: str,
    input_snapshot: Mapping[str, Any],
    output_allocations: Mapping[str, Any],
) -> dict[str, Any]:
    envelope = {
        "protocol_version": PROTOCOL_VERSION,
        "job_id": job_id,
        "instance_id": instance_id,
        "spawn_nonce": spawn_nonce,
        "schema_id": schema_id,
        "request_sha256": request_sha256,
        "request": dict(request),
        "input_snapshot_sha256": input_snapshot_sha256,
        "input_snapshot": dict(input_snapshot),
        "output_allocations": dict(output_allocations),
    }
    return dict(validate_request_envelope(envelope, job_id=job_id, instance_id=instance_id))


def validate_request_envelope(
    value: object, *, job_id: str, instance_id: str
) -> Mapping[str, Any]:
    envelope = _exact(
        value,
        {
            "protocol_version",
            "job_id",
            "instance_id",
            "spawn_nonce",
            "schema_id",
            "request_sha256",
            "request",
            "input_snapshot_sha256",
            "input_snapshot",
            "output_allocations",
        },
        "request envelope",
    )
    if envelope["protocol_version"] != PROTOCOL_VERSION:
        raise WorkerProtocolError("unsupported worker protocol version")
    if _canonical_uuid(envelope["job_id"], "job_id") != _canonical_uuid(job_id, "job_id"):
        raise WorkerProtocolError("request job identity does not match argv")
    if _canonical_uuid(envelope["instance_id"], "instance_id") != _canonical_uuid(
        instance_id, "instance_id"
    ):
        raise WorkerProtocolError("request instance identity does not match argv")
    _digest(envelope["spawn_nonce"], "spawn_nonce")
    _bounded_text(envelope["schema_id"], "schema_id", 1, 200)
    request_digest = _digest(envelope["request_sha256"], "request_sha256")
    request = envelope["request"]
    if not isinstance(request, Mapping):
        raise WorkerProtocolError("request must be an object")
    if request.get("operation") not in OPERATIONS:
        raise WorkerProtocolError("request operation is outside the closed dispatch map")
    if hashlib.sha256(canonical_json(dict(request))).hexdigest() != request_digest:
        raise WorkerProtocolError("request digest does not match canonical request bytes")
    snapshot_digest = _digest(envelope["input_snapshot_sha256"], "input_snapshot_sha256")
    snapshot = validate_input_snapshot(envelope["input_snapshot"])
    if hashlib.sha256(canonical_json(dict(snapshot))).hexdigest() != snapshot_digest:
        raise WorkerProtocolError("input snapshot digest does not match canonical bytes")
    allocations = validate_output_allocations(envelope["output_allocations"])
    if dict(allocations) != dict(snapshot["ids"]):
        raise WorkerProtocolError("output allocations do not match the input snapshot")
    if snapshot["job_id"] != envelope["job_id"]:
        raise WorkerProtocolError("input snapshot job identity does not match")
    if snapshot["operation"] != request.get("operation"):
        raise WorkerProtocolError("input snapshot operation does not match")
    if snapshot["request_sha256"] != request_digest:
        raise WorkerProtocolError("input snapshot request digest does not match")
    return envelope


def validate_event_payload(event_type: str, value: object) -> Mapping[str, Any]:
    if event_type not in _EVENT_TYPES:
        raise WorkerProtocolError("unknown worker event type")
    if event_type == "state_changed":
        payload = _exact(value, {"state", "step", "requested_final_step"}, "state payload")
        if payload["state"] not in _STATES:
            raise WorkerProtocolError("unknown job state")
        for name in ("step", "requested_final_step"):
            if payload[name] is not None:
                _bounded_int(payload[name], name)
        return payload
    if event_type == "phase_changed":
        payload = _exact(value, {"phase"}, "phase payload")
        if payload["phase"] not in _PHASES:
            raise WorkerProtocolError("unknown job phase")
        return payload
    if event_type == "progress":
        payload = _exact(value, {"current", "total", "unit", "message"}, "progress payload")
        current = _bounded_int(payload["current"], "current")
        total = payload["total"]
        if total is not None:
            total = _bounded_int(total, "total")
            if current > total:
                raise WorkerProtocolError("progress current exceeds total")
        if payload["unit"] not in _PROGRESS_UNITS:
            raise WorkerProtocolError("unknown progress unit")
        _bounded_text(payload["message"], "message", 0, 500)
        return payload
    if event_type == "metric":
        payload = _exact(value, {"phase", "step", "split", "name", "value"}, "metric payload")
        _bounded_text(payload["phase"], "phase", 1, 100)
        _bounded_int(payload["step"], "step")
        if payload["split"] not in {"train", "validation", "test"}:
            raise WorkerProtocolError("unknown metric split")
        _bounded_text(payload["name"], "name", 1, 100)
        metric = payload["value"]
        if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(metric):
            raise WorkerProtocolError("metric value must be finite")
        return payload
    if event_type == "warning":
        payload = _exact(value, {"code", "message"}, "warning payload")
        _bounded_text(payload["code"], "code", 1, 100)
        _bounded_text(payload["message"], "message", 1, 2_000)
        return payload
    if event_type == "checkpoint_committed":
        payload = _exact(value, {"checkpoint_id", "step", "sha256"}, "checkpoint payload")
        _canonical_uuid(payload["checkpoint_id"], "checkpoint_id")
        _bounded_int(payload["step"], "step")
        _digest(payload["sha256"], "sha256")
        return payload
    payload = _exact(
        value,
        {"state", "reason_code", "checkpoint_id", "checkpoint_step"},
        "terminal payload",
    )
    if payload["state"] not in _TERMINAL_STATES:
        raise WorkerProtocolError("terminal payload has a nonterminal state")
    reason = payload["reason_code"]
    if reason not in _TERMINAL_REASONS and reason not in _JOB_ERROR_CODES:
        raise WorkerProtocolError("terminal reason is outside the closed vocabulary")
    if payload["state"] == "completed" and reason != "completed":
        raise WorkerProtocolError("completed terminal payload has the wrong reason")
    checkpoint_id, checkpoint_step = payload["checkpoint_id"], payload["checkpoint_step"]
    if (checkpoint_id is None) != (checkpoint_step is None):
        raise WorkerProtocolError("terminal checkpoint fields must be paired")
    if checkpoint_id is not None:
        _canonical_uuid(checkpoint_id, "checkpoint_id")
        _bounded_int(checkpoint_step, "checkpoint_step")
    return payload


def validate_worker_message(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not isinstance(value.get("type"), str):
        raise WorkerProtocolError("worker message has no type")
    kind = value["type"]
    if kind == "ready":
        message = _exact(
            value,
            {
                "type",
                "protocol_version",
                "job_id",
                "instance_id",
                "spawn_nonce",
                "request_sha256",
                "schema_id",
                "input_snapshot_sha256",
            },
            "ready message",
        )
        if message["protocol_version"] != PROTOCOL_VERSION:
            raise WorkerProtocolError("worker acknowledged the wrong protocol version")
        _canonical_uuid(message["job_id"], "job_id")
        _canonical_uuid(message["instance_id"], "instance_id")
        _digest(message["spawn_nonce"], "spawn_nonce")
        _digest(message["request_sha256"], "request_sha256")
        _digest(message["input_snapshot_sha256"], "input_snapshot_sha256")
        _bounded_text(message["schema_id"], "schema_id", 1, 200)
        return message
    if kind == "event":
        message = _exact(value, {"type", "event_type", "payload"}, "event message")
        validate_event_payload(message["event_type"], message["payload"])
        if message["event_type"] in {"state_changed", "terminal"}:
            raise WorkerProtocolError("the scheduler owns state and terminal events")
        if message["event_type"] == "checkpoint_committed":
            raise WorkerProtocolError("the scheduler owns checkpoint committed events")
        return message
    if kind == "artifact_ready":
        message = _exact(
            value,
            {"type", "role", "staging_name", "size_bytes", "sha256"},
            "artifact_ready message",
        )
        if message["role"] not in ARTIFACT_ROLES:
            raise WorkerProtocolError("artifact role is outside the closed vocabulary")
        _basename(message["staging_name"], "staging_name")
        _bounded_int(message["size_bytes"], "size_bytes", maximum=1 << 40)
        _digest(message["sha256"], "sha256")
        return message
    if kind == "checkpoint_ready":
        message = _exact(
            value,
            {"type", "staging_name", "manifest_sha256", "files"},
            "checkpoint_ready message",
        )
        _basename(message["staging_name"], "staging_name")
        _digest(message["manifest_sha256"], "manifest_sha256")
        files = message["files"]
        if not isinstance(files, list) or len(files) != 7:
            raise WorkerProtocolError("checkpoint_ready must describe exactly seven files")
        names: set[str] = set()
        for index, raw in enumerate(files):
            item = _exact(raw, {"name", "size", "sha256"}, f"checkpoint files[{index}]")
            name = _basename(item["name"], "checkpoint file name")
            if name in names:
                raise WorkerProtocolError("checkpoint file names must be unique")
            names.add(name)
            _bounded_int(item["size"], "checkpoint file size", maximum=1 << 40)
            _digest(item["sha256"], "checkpoint file sha256")
        required = {
            "model.safetensors",
            "optimizer.safetensors",
            "rng.safetensors",
            "tokenizer.json",
            "config.json",
            "trainer_state.json",
            "manifest.json",
        }
        if names != required:
            raise WorkerProtocolError("checkpoint_ready has the wrong file set")
        manifest = next(item for item in files if item["name"] == "manifest.json")
        if manifest["sha256"] != message["manifest_sha256"]:
            raise WorkerProtocolError("checkpoint manifest digest is inconsistent")
        return message
    if kind == "result":
        message = _exact(value, {"type", "operation", "result"}, "result message")
        if message["operation"] not in OPERATIONS or not isinstance(message["result"], Mapping):
            raise WorkerProtocolError("worker result is not operation-discriminated")
        return message
    if kind == "interrupted":
        message = _exact(
            value,
            {"type", "reason_code", "checkpoint_id", "checkpoint_step", "error"},
            "interrupted message",
        )
        if message["reason_code"] not in _TERMINAL_REASONS - {"completed"}:
            raise WorkerProtocolError("unknown interruption reason")
        if (message["checkpoint_id"] is None) != (message["checkpoint_step"] is None):
            raise WorkerProtocolError("interrupted checkpoint fields must be paired")
        if message["checkpoint_id"] is not None:
            _canonical_uuid(message["checkpoint_id"], "checkpoint_id")
            _bounded_int(message["checkpoint_step"], "checkpoint_step")
        if message["error"] is not None:
            validate_job_error(message["error"], interruption=True)
        return message
    if kind == "error":
        message = _exact(value, {"type", "error"}, "error message")
        validate_job_error(message["error"])
        return message
    raise WorkerProtocolError("unknown worker message type")


def validate_parent_message(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not isinstance(value.get("type"), str):
        raise WorkerProtocolError("parent message has no type")
    kind = value["type"]
    if kind == "artifact_prepared":
        message = _exact(
            value,
            {"type", "role", "artifact_id", "sha256"},
            "artifact prepared acknowledgment",
        )
        if message["role"] not in ARTIFACT_ROLES:
            raise WorkerProtocolError("artifact acknowledgment role is invalid")
        _canonical_uuid(message["artifact_id"], "artifact_id")
        _digest(message["sha256"], "sha256")
        return message
    if kind == "checkpoint_committed":
        message = _exact(
            value,
            {
                "type",
                "checkpoint_id",
                "step",
                "sha256",
                "run_id",
                "artifact_ids",
            },
            "checkpoint committed acknowledgment",
        )
        _canonical_uuid(message["checkpoint_id"], "checkpoint_id")
        _bounded_int(message["step"], "step")
        _digest(message["sha256"], "sha256")
        _canonical_uuid(message["run_id"], "run_id")
        artifact_ids = message["artifact_ids"]
        if not isinstance(artifact_ids, Mapping) or len(artifact_ids) != 7:
            raise WorkerProtocolError("checkpoint artifact_ids is invalid")
        required = {
            "model.safetensors",
            "optimizer.safetensors",
            "rng.safetensors",
            "tokenizer.json",
            "config.json",
            "trainer_state.json",
            "manifest.json",
        }
        if set(artifact_ids) != required:
            raise WorkerProtocolError("checkpoint artifact_ids has the wrong file set")
        for name, artifact_id in artifact_ids.items():
            _basename(name, "checkpoint artifact name")
            _canonical_uuid(artifact_id, "checkpoint artifact_id")
        return message
    if kind == "commit_rejected":
        message = _exact(value, {"type", "kind", "error"}, "commit rejection")
        if message["kind"] not in {"artifact", "checkpoint"}:
            raise WorkerProtocolError("commit rejection kind is invalid")
        validate_job_error(message["error"])
        return message
    raise WorkerProtocolError("unknown parent message type")


def validate_job_error(value: object, *, interruption: bool = False) -> Mapping[str, Any]:
    error = _exact(value, {"code", "message", "retryable", "field_errors"}, "job error")
    allowed = {"CHECKPOINT_WRITE_FAILED", "WORKER_OWNERSHIP_UNKNOWN"} if interruption else _JOB_ERROR_CODES
    if error["code"] not in allowed:
        raise WorkerProtocolError("job error code is outside the closed vocabulary")
    _bounded_text(error["message"], "message", 1, 2_000)
    if not isinstance(error["retryable"], bool):
        raise WorkerProtocolError("job error retryable must be boolean")
    field_errors = error["field_errors"]
    if not isinstance(field_errors, list) or len(field_errors) > 100:
        raise WorkerProtocolError("job error field_errors is invalid")
    for index, item in enumerate(field_errors):
        item = _exact(item, {"field_path", "message"}, f"field_errors[{index}]")
        _bounded_text(item["field_path"], "field_path", 1, 500)
        _bounded_text(item["message"], "message", 1, 1_000)
    return error


@dataclass
class ProtocolState:
    """Validate identity and ordering across messages from one worker."""

    job_id: str
    instance_id: str
    spawn_nonce: str
    request_sha256: str
    schema_id: str
    operation: str
    input_snapshot_sha256: str | None = None
    ready: bool = False
    terminal: bool = False
    awaiting_ack: str | None = None

    def accept(self, raw: object) -> Mapping[str, Any]:
        message = validate_worker_message(raw)
        if self.terminal:
            raise WorkerProtocolError("worker emitted data after a terminal message")
        if not self.ready:
            if message["type"] != "ready":
                raise WorkerProtocolError("worker did not acknowledge before producing output")
            expected = {
                "job_id": self.job_id,
                "instance_id": self.instance_id,
                "spawn_nonce": self.spawn_nonce,
                "request_sha256": self.request_sha256,
                "schema_id": self.schema_id,
                "input_snapshot_sha256": self.input_snapshot_sha256,
            }
            if any(message[name] != value for name, value in expected.items()):
                raise WorkerProtocolError("worker acknowledgment identity does not match dispatch")
            self.ready = True
            return message
        if message["type"] == "ready":
            raise WorkerProtocolError("worker acknowledged more than once")
        if self.awaiting_ack is not None:
            raise WorkerProtocolError("worker emitted data before parent commit acknowledgment")
        if message["type"] in {"artifact_ready", "checkpoint_ready"}:
            self.awaiting_ack = "artifact" if message["type"] == "artifact_ready" else "checkpoint"
            return message
        if message["type"] == "result":
            if message["operation"] != self.operation:
                raise WorkerProtocolError("worker result operation does not match request")
            self.terminal = True
        elif message["type"] in {"error", "interrupted"}:
            self.terminal = True
        return message

    def acknowledge(self, raw: object) -> Mapping[str, Any]:
        message = validate_parent_message(raw)
        if self.awaiting_ack is None:
            raise WorkerProtocolError("parent acknowledgment has no pending proposal")
        if message["type"] == "commit_rejected":
            kind = message["kind"]
        else:
            kind = "artifact" if message["type"] == "artifact_prepared" else "checkpoint"
        if kind != self.awaiting_ack:
            raise WorkerProtocolError("parent acknowledgment kind does not match proposal")
        self.awaiting_ack = None
        return message


class CancellationToken:
    """Worker-side cancellation reader that never polls parent process state."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._event = threading.Event()
        self._reason: str | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._read, name="worker-cancel", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        try:
            message = read_frame(self._stream, allow_eof=True)
            if message is None:
                return
            parsed = _exact(message, {"reason_code"}, "cancellation request")
            reason = parsed["reason_code"]
            if reason not in {"user_cancelled", "timeout", "service_shutdown"}:
                return
            with self._lock:
                self._reason = reason
                self._event.set()
        except (EOFError, OSError, WorkerProtocolError):
            return

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)


def open_inherited_stream(variable: str, mode: str) -> BinaryIO:
    raw = os.environ.pop(variable, None)
    if raw is None:
        raise WorkerProtocolError(f"missing inherited pipe {variable}")
    if os.name == "nt":
        if re.fullmatch(r"h:[0-9]+", raw) is None:
            raise WorkerProtocolError(f"invalid inherited pipe {variable}")
        import msvcrt

        flags = os.O_RDONLY if "r" in mode else os.O_WRONLY
        fd = msvcrt.open_osfhandle(int(raw[2:]), flags)
    else:
        if re.fullmatch(r"[0-9]+", raw) is None:
            raise WorkerProtocolError(f"invalid inherited pipe {variable}")
        fd = int(raw)
    if fd < 3:
        raise WorkerProtocolError(f"invalid inherited pipe {variable}")
    return os.fdopen(fd, mode, buffering=0)


__all__ = [
    "ARTIFACT_ROLES",
    "CancellationToken",
    "MAX_FRAME_BYTES",
    "PROTOCOL_VERSION",
    "ProtocolState",
    "WorkerProtocolError",
    "encode_frame",
    "make_request_envelope",
    "open_inherited_stream",
    "read_frame",
    "validate_event_payload",
    "validate_job_error",
    "validate_input_snapshot",
    "validate_output_allocations",
    "validate_parent_message",
    "validate_request_envelope",
    "validate_worker_message",
    "write_frame",
]
