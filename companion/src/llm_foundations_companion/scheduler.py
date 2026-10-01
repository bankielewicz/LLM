"""FIFO job scheduler, idempotency store, and owned worker processes."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import queue
import secrets
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, BinaryIO

from .errors import ApiError, ERROR_VOCABULARY
from .events import EventLog, EventPage
from .operations import OPERATIONS
from .schema import canonical_json, strict_json, validate_schema
from .worker_protocol import (
    ProtocolState,
    WorkerProtocolError,
    make_request_envelope,
    read_frame,
    write_frame,
)


SCHEDULER_SCHEMA_VERSION = 1
MAX_QUEUED_JOBS = 8
DEADLINE_SECONDS = 3_600.0
CANCEL_GRACE_SECONDS = 30.0
ROOT_QUOTA_BYTES = 53_687_091_200

SCHEDULER_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        state TEXT NOT NULL CHECK (
            state IN ('queued','starting','running','cancelling','completed','failed','interrupted')
        ),
        operation TEXT NOT NULL,
        request_sha256 TEXT NOT NULL CHECK (length(request_sha256) = 64),
        idempotency_key TEXT,
        request_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
        request_schema_id TEXT NOT NULL,
        reservation_id TEXT NOT NULL UNIQUE,
        spawn_nonce TEXT,
        worker_pid INTEGER,
        cancel_requested_at TEXT,
        cancel_reason TEXT,
        record_json TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS jobs_created_idx
    ON jobs(created_at DESC, job_id DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS jobs_fifo_idx
    ON jobs(state, created_at ASC, job_id ASC)
    """,
    """
    CREATE INDEX IF NOT EXISTS jobs_operation_idx
    ON jobs(operation, created_at DESC, job_id DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS jobs_request_idx
    ON jobs(request_sha256, created_at DESC, job_id DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS jobs_idempotency_idx
    ON jobs(idempotency_key, created_at DESC, job_id DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS job_events (
        job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE RESTRICT,
        cursor INTEGER NOT NULL CHECK (cursor BETWEEN 1 AND 2147483647),
        event_type TEXT NOT NULL CHECK (
            event_type IN ('state_changed','phase_changed','progress','metric','warning','checkpoint_committed','terminal')
        ),
        occurred_at TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        PRIMARY KEY(job_id, cursor)
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS job_events_no_update
    BEFORE UPDATE ON job_events
    BEGIN SELECT RAISE(ABORT, 'job events are immutable'); END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS job_events_no_delete
    BEFORE DELETE ON job_events
    BEGIN SELECT RAISE(ABORT, 'job events are immutable'); END
    """,
    """
    CREATE TABLE IF NOT EXISTS idempotency_records (
        instance_id TEXT NOT NULL,
        method TEXT NOT NULL,
        resolved_path TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        request_sha256 TEXT NOT NULL CHECK (length(request_sha256) = 64),
        status_code INTEGER NOT NULL CHECK (status_code BETWEEN 200 AND 299),
        response_body BLOB NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        PRIMARY KEY(instance_id, method, resolved_path, idempotency_key)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idempotency_expiry_idx
    ON idempotency_records(expires_at)
    """,
)

REQUEST_SCHEMAS = {
    "tokenizer_train": "TokenizerTrainRequest",
    "tiny_train": "TinyTrainRequest",
    "tiny_resume": "TinyResumeRequest",
    "evaluate": "EvaluateRequest",
    "generate": "GenerateRequest",
    "model_prepare": "ModelPrepareRequest",
    "adapter_train": "AdapterTrainRequest",
    "adapter_resume": "AdapterResumeRequest",
    "context_preview": "ContextPreviewRequest",
    "chat_generate": "ChatGenerateRequest",
    "retrieval_build": "RetrievalBuildRequest",
    "retrieval_query": "RetrievalQueryRequest",
    "export_bundle": "ExportBundleRequest",
    "validate_bundle": "ValidateBundleRequest",
    "import_bundle": "ImportBundleRequest",
}

_TERMINAL = frozenset({"completed", "failed", "interrupted"})
_ACTIVE = frozenset({"starting", "running", "cancelling"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _canonical_uuid(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a lowercase UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"{field} must be a lowercase UUID") from exc
    if str(parsed) != value:
        raise ValueError(f"{field} must be a lowercase UUID")
    return value


def _as_bytes(value: bytes | bytearray | memoryview | Mapping[str, Any]) -> bytes:
    if isinstance(value, Mapping):
        return canonical_json(dict(value))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError("response body must be bytes or a JSON object")


@dataclass(frozen=True)
class StoredResponse:
    status_code: int
    response_body: bytes


@dataclass(frozen=True)
class Submission:
    status_code: int
    response_body: bytes
    replayed: bool

    @property
    def job(self) -> dict[str, Any]:
        value = strict_json(self.response_body)
        if not isinstance(value, dict):
            raise RuntimeError("stored submission response is not an object")
        return value


@dataclass(frozen=True)
class IdempotencyCommit:
    store: "IdempotencyStore"
    method: str
    resolved_path: str
    key: str
    request_sha256: str

    def record_success(
        self,
        connection: sqlite3.Connection,
        *,
        status_code: int,
        response_body: bytes | bytearray | memoryview | Mapping[str, Any],
    ) -> bytes:
        return self.store.record_success(
            connection,
            method=self.method,
            resolved_path=self.resolved_path,
            key=self.key,
            request_sha256=self.request_sha256,
            status_code=status_code,
            response_body=response_body,
        )


class IdempotencyStore:
    """Twenty-four-hour instance/method/resolved-route replay records."""

    def __init__(self, database: Any, instance_id: str, *, clock: Callable[[], Any] | None = None) -> None:
        self.database = database
        self.instance_id = _canonical_uuid(instance_id, "instance_id")
        self._clock = clock

    def _now(self) -> datetime:
        if self._clock is None:
            return datetime.now(timezone.utc)
        value = self._clock()
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str):
            return _parse_time(value)
        raise TypeError("clock must return datetime or RFC 3339 text")

    @staticmethod
    def _scope(method: str, resolved_path: str, key: str, request_sha256: str) -> tuple[str, str, str, str]:
        if method not in {"POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError("idempotency method must be a mutation")
        if not isinstance(resolved_path, str) or not resolved_path.startswith("/api/v1/"):
            raise ValueError("idempotency path must be a resolved API path")
        _canonical_uuid(key, "idempotency key")
        if not isinstance(request_sha256, str) or len(request_sha256) != 64:
            raise ValueError("request digest is invalid")
        try:
            int(request_sha256, 16)
        except ValueError as exc:
            raise ValueError("request digest is invalid") from exc
        if request_sha256 != request_sha256.lower():
            raise ValueError("request digest is invalid")
        return method, resolved_path, key, request_sha256

    def prepare(self, method: str, resolved_path: str, key: str, request_sha256: str) -> IdempotencyCommit:
        method, resolved_path, key, request_sha256 = self._scope(
            method, resolved_path, key, request_sha256
        )
        return IdempotencyCommit(self, method, resolved_path, key, request_sha256)

    def lookup(
        self, method: str, resolved_path: str, key: str, request_sha256: str
    ) -> StoredResponse | None:
        with self.database.read() as connection:
            return self.lookup_in(connection, method, resolved_path, key, request_sha256)

    def lookup_in(
        self,
        connection: sqlite3.Connection,
        method: str,
        resolved_path: str,
        key: str,
        request_sha256: str,
    ) -> StoredResponse | None:
        method, resolved_path, key, request_sha256 = self._scope(
            method, resolved_path, key, request_sha256
        )
        row = connection.execute(
            """
            SELECT request_sha256, status_code, response_body, expires_at
            FROM idempotency_records
            WHERE instance_id = ? AND method = ? AND resolved_path = ? AND idempotency_key = ?
            """,
            (self.instance_id, method, resolved_path, key),
        ).fetchone()
        if row is None or _parse_time(str(row[3])) <= self._now():
            return None
        if str(row[0]) != request_sha256:
            raise ApiError("IDEMPOTENCY_CONFLICT")
        return StoredResponse(int(row[1]), bytes(row[2]))

    def record_success(
        self,
        connection: sqlite3.Connection,
        *,
        method: str,
        resolved_path: str,
        key: str,
        request_sha256: str,
        status_code: int,
        response_body: bytes | bytearray | memoryview | Mapping[str, Any],
    ) -> bytes:
        method, resolved_path, key, request_sha256 = self._scope(
            method, resolved_path, key, request_sha256
        )
        if isinstance(status_code, bool) or not isinstance(status_code, int) or not 200 <= status_code <= 299:
            raise ValueError("only 2xx idempotency responses may be stored")
        body = _as_bytes(response_body)
        now = self._now()
        created_at = now.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        expires_at = (now + timedelta(hours=24)).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        )
        existing = connection.execute(
            """
            SELECT request_sha256, status_code, response_body, expires_at
            FROM idempotency_records
            WHERE instance_id = ? AND method = ? AND resolved_path = ? AND idempotency_key = ?
            """,
            (self.instance_id, method, resolved_path, key),
        ).fetchone()
        if existing is not None and _parse_time(str(existing[3])) > now:
            if str(existing[0]) != request_sha256:
                raise ApiError("IDEMPOTENCY_CONFLICT")
            if int(existing[1]) != status_code or bytes(existing[2]) != body:
                raise RuntimeError("idempotency success changed inside its retention window")
            return bytes(existing[2])
        connection.execute(
            """
            INSERT INTO idempotency_records(
                instance_id, method, resolved_path, idempotency_key, request_sha256,
                status_code, response_body, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(instance_id, method, resolved_path, idempotency_key)
            DO UPDATE SET request_sha256=excluded.request_sha256,
                          status_code=excluded.status_code,
                          response_body=excluded.response_body,
                          created_at=excluded.created_at,
                          expires_at=excluded.expires_at
            """,
            (
                self.instance_id,
                method,
                resolved_path,
                key,
                request_sha256,
                status_code,
                body,
                created_at,
                expires_at,
            ),
        )
        return body


class AdmissionPlanner:
    """Checked RUN-007 byte and row reservations for installed operations."""

    def plan(self, request: Mapping[str, Any]) -> dict[str, int]:
        operation = request["operation"]
        if operation == "tokenizer_train":
            byte_count = 64 * 1024 * 1024
        elif operation == "model_prepare":
            byte_count = 272_437_573 + 512 * 1024 * 1024
        elif operation in {
            "evaluate",
            "generate",
            "context_preview",
            "chat_generate",
            "retrieval_build",
            "retrieval_query",
        }:
            byte_count = 256 * 1024 * 1024
        elif operation == "validate_bundle":
            byte_count = 64 * 1024 * 1024
        elif operation in {"tiny_train", "tiny_resume", "adapter_train", "adapter_resume", "export_bundle", "import_bundle"}:
            raise ApiError(
                "CAPABILITY_UNAVAILABLE",
                "This operation requires an installed admission planner.",
            )
        else:
            raise ApiError("CAPABILITY_UNAVAILABLE")
        if byte_count > ROOT_QUOTA_BYTES:
            raise ApiError(
                "VALIDATION_FAILED", reason_code="ESTIMATE_EXCEEDS_ROOT_QUOTA"
            )
        return {
            "byte_count": byte_count,
            "artifact_rows": 16,
            "dataset_rows": 0,
            "run_rows": 1,
            "model_rows": 1,
            "checkpoint_rows": 0,
        }


class _BoundedCapture:
    def __init__(self, stream: BinaryIO, limit: int = 1_048_576) -> None:
        self.data = bytearray()
        self.total_bytes = 0
        self.truncated = False
        self._stream = stream
        self._limit = limit
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        try:
            while chunk := self._stream.read(65_536):
                self.total_bytes += len(chunk)
                room = self._limit - len(self.data)
                if room > 0:
                    self.data.extend(chunk[:room])
                if len(chunk) > room:
                    self.truncated = True
        finally:
            try:
                self._stream.close()
            except OSError:
                pass

    def join(self) -> None:
        self._thread.join(timeout=1)


class OwnedWorkerProcess:
    """One retained process-group/Job-Object identity and its private pipes."""

    def __init__(
        self,
        *,
        process: subprocess.Popen[bytes],
        job_id: str,
        spawn_nonce: str,
        request_writer: BinaryIO,
        cancel_writer: BinaryIO,
        protocol_reader: BinaryIO,
        protocol_state: ProtocolState,
        windows_job: int | None,
    ) -> None:
        self.process = process
        self.job_id = job_id
        self.spawn_nonce = spawn_nonce
        self._request_writer = request_writer
        self._cancel_writer = cancel_writer
        self._protocol_reader = protocol_reader
        self._protocol_state = protocol_state
        self._windows_job = windows_job
        self._container_terminated = False
        self._messages: queue.Queue[Mapping[str, Any] | BaseException | None] = queue.Queue()
        self._cancel_reason: str | None = None
        self._protocol_eof = False
        self.stdout = _BoundedCapture(process.stdout) if process.stdout is not None else None
        self.stderr = _BoundedCapture(process.stderr) if process.stderr is not None else None
        self._reader = threading.Thread(target=self._read_protocol, daemon=True)
        self._reader.start()

    def send_request(self, envelope: Mapping[str, Any]) -> None:
        write_frame(self._request_writer, envelope)
        self._request_writer.close()

    def _read_protocol(self) -> None:
        try:
            while True:
                raw = read_frame(self._protocol_reader, allow_eof=True)
                if raw is None:
                    self._protocol_eof = True
                    self._messages.put(None)
                    return
                self._messages.put(self._protocol_state.accept(raw))
        except BaseException as exc:
            self._messages.put(exc)
        finally:
            try:
                self._protocol_reader.close()
            except OSError:
                pass

    def messages(self) -> list[Mapping[str, Any] | BaseException | None]:
        result: list[Mapping[str, Any] | BaseException | None] = []
        while True:
            try:
                result.append(self._messages.get_nowait())
            except queue.Empty:
                return result

    @property
    def ready(self) -> bool:
        return self._protocol_state.ready

    @property
    def terminal(self) -> bool:
        return self._protocol_state.terminal

    def poll(self) -> int | None:
        return self.process.poll()

    @property
    def protocol_eof(self) -> bool:
        return self._protocol_eof

    def request_cancel(self, reason: str) -> None:
        if self._cancel_reason is not None:
            return
        if reason not in {"user_cancelled", "timeout", "service_shutdown"}:
            raise ValueError("unsupported worker cancellation reason")
        write_frame(self._cancel_writer, {"reason_code": reason})
        self._cancel_writer.close()
        self._cancel_reason = reason

    def terminate_owned(self, expected_nonce: str) -> None:
        if not secrets.compare_digest(self.spawn_nonce, expected_nonce):
            raise WorkerProtocolError("owned worker nonce does not match")
        if self._container_terminated:
            return
        if os.name == "nt":
            if self._windows_job is None:
                raise WorkerProtocolError("Windows worker has no retained Job Object")
            if not ctypes.windll.kernel32.TerminateJobObject(self._windows_job, 1):
                raise OSError("TerminateJobObject failed")
        else:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                if self.poll() is None:
                    raise
        if self.poll() is None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired as exc:
                raise OSError("owned worker leader did not exit") from exc
        self._container_terminated = True

    def close(self) -> None:
        failure: OSError | WorkerProtocolError | None = None
        try:
            self.terminate_owned(self.spawn_nonce)
        except (OSError, WorkerProtocolError) as exc:
            failure = exc
        for stream in (self._request_writer, self._cancel_writer):
            try:
                stream.close()
            except OSError:
                pass
        self._reader.join(timeout=1)
        if self.stdout is not None:
            self.stdout.join()
        if self.stderr is not None:
            self.stderr.join()
        if self._windows_job is not None:
            ctypes.windll.kernel32.CloseHandle(self._windows_job)
            self._windows_job = None
        if failure is not None:
            raise failure


def _assign_windows_job(process: subprocess.Popen[bytes]) -> int:
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.SetInformationJobObject.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.TerminateJobObject.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class BASIC_LIMIT(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BASIC_LIMIT),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        raise OSError("CreateJobObjectW failed")
    info = EXTENDED_LIMIT()
    info.BasicLimitInformation.LimitFlags = 0x00002000
    if not kernel32.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
        kernel32.CloseHandle(handle)
        raise OSError("SetInformationJobObject failed")
    if not kernel32.AssignProcessToJobObject(handle, ctypes.c_void_p(int(process._handle))):
        kernel32.CloseHandle(handle)
        raise OSError("AssignProcessToJobObject failed")
    return int(handle)


class WorkerController:
    """Production launcher with fixed argv, cwd, environment, and ownership."""

    def __init__(self, *, instance_lease: Any = None) -> None:
        self.instance_lease = instance_lease

    @property
    def command_prefix(self) -> tuple[str, ...]:
        return (sys.executable, "-I", "-m", "llm_foundations_companion.worker_main")

    def launch(
        self,
        *,
        job_id: str,
        instance_id: str,
        runtime_profile: str,
        root: Path,
        request: Mapping[str, Any],
        request_sha256: str,
        schema_id: str,
    ) -> OwnedWorkerProcess:
        spawn_nonce = secrets.token_hex(32)
        request_read, request_write = os.pipe()
        protocol_read, protocol_write = os.pipe()
        cancel_read, cancel_write = os.pipe()
        child_fds = [request_read, protocol_write, cancel_read]
        lease_fd: int | None = None
        try:
            staging = root / "jobs" / job_id / "staging"
            staging.mkdir(mode=0o700, parents=True, exist_ok=True)
            if staging.is_symlink() or not staging.is_dir() or staging.resolve() != staging:
                raise OSError("worker staging directory is not confined")
            if any(staging.iterdir()):
                raise OSError("worker staging directory is not empty")
            if self.instance_lease is not None:
                lease_fd = int(self.instance_lease.make_inheritable())
                child_fds.append(lease_fd)
            if os.name == "nt":
                import msvcrt

                child_handles = [msvcrt.get_osfhandle(fd) for fd in child_fds]
                for handle in child_handles:
                    os.set_handle_inheritable(handle, True)
                pipe_values = {
                    "LLMF_REQUEST_FD": f"h:{child_handles[0]}",
                    "LLMF_PROTOCOL_FD": f"h:{child_handles[1]}",
                    "LLMF_CANCEL_FD": f"h:{child_handles[2]}",
                }
            else:
                for fd in child_fds:
                    os.set_inheritable(fd, True)
                child_handles = []
                pipe_values = {
                    "LLMF_REQUEST_FD": str(request_read),
                    "LLMF_PROTOCOL_FD": str(protocol_write),
                    "LLMF_CANCEL_FD": str(cancel_read),
                }
            environment = {
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                **pipe_values,
                "LLMF_JOB_ID": job_id,
                "LLMF_INSTANCE_ID": instance_id,
                "LLMF_SPAWN_NONCE": spawn_nonce,
                "LLMF_PARENT_PID": str(os.getpid()),
                "LLMF_RUNTIME_PROFILE": runtime_profile,
                "LLMF_STAGING_PATH": str(staging),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            }
            if os.name == "nt":
                for name in ("SYSTEMROOT", "WINDIR"):
                    if name in os.environ:
                        environment[name] = os.environ[name]
            command = (*self.command_prefix, "--job-id", job_id, "--instance-id", instance_id)
        except BaseException:
            if lease_fd is not None:
                os.set_inheritable(lease_fd, False)
            for fd in (request_read, request_write, protocol_read, protocol_write, cancel_read, cancel_write):
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise
        process: subprocess.Popen[bytes] | None = None
        windows_job: int | None = None
        try:
            kwargs: dict[str, Any] = {
                "cwd": str(Path(__file__).resolve().parents[1]),
                "env": environment,
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "close_fds": os.name != "nt",
            }
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.lpAttributeList = {"handle_list": child_handles}
                kwargs["startupinfo"] = startupinfo
                kwargs["close_fds"] = True
            else:
                kwargs["pass_fds"] = tuple(child_fds)
                kwargs["start_new_session"] = True
            process = subprocess.Popen(command, **kwargs)
            if lease_fd is not None:
                os.set_inheritable(lease_fd, False)
            if os.name == "nt":
                windows_job = _assign_windows_job(process)
            for fd in (request_read, protocol_write, cancel_read):
                os.close(fd)
            owned = OwnedWorkerProcess(
                process=process,
                job_id=job_id,
                spawn_nonce=spawn_nonce,
                request_writer=os.fdopen(request_write, "wb", buffering=0),
                cancel_writer=os.fdopen(cancel_write, "wb", buffering=0),
                protocol_reader=os.fdopen(protocol_read, "rb", buffering=0),
                protocol_state=ProtocolState(
                    job_id=job_id,
                    instance_id=instance_id,
                    spawn_nonce=spawn_nonce,
                    request_sha256=request_sha256,
                    schema_id=schema_id,
                    operation=str(request["operation"]),
                ),
                windows_job=windows_job,
            )
            owned.send_request(
                make_request_envelope(
                    job_id=job_id,
                    instance_id=instance_id,
                    spawn_nonce=spawn_nonce,
                    schema_id=schema_id,
                    request_sha256=request_sha256,
                    request=request,
                )
            )
            return owned
        except BaseException:
            if lease_fd is not None:
                os.set_inheritable(lease_fd, False)
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            if windows_job is not None:
                ctypes.windll.kernel32.CloseHandle(windows_job)
            for fd in (request_read, request_write, protocol_read, protocol_write, cancel_read, cancel_write):
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise


class Scheduler:
    def __init__(
        self,
        database: Any,
        registry: Any,
        root: Path,
        instance_id: str,
        installation_id: str,
        runtime_profile: str,
        app_version: str,
        capabilities: Mapping[str, Any],
        admission_planner: Any = None,
        worker_controller: Any = None,
        clock: Callable[[], Any] | None = None,
        uuid_factory: Callable[[], Any] | None = None,
        instance_lease: Any = None,
    ) -> None:
        self.database = database
        self.registry = registry
        self.root = Path(root).resolve()
        self.instance_id = _canonical_uuid(instance_id, "instance_id")
        self.installation_id = _canonical_uuid(installation_id, "installation_id")
        self.runtime_profile = runtime_profile
        self.app_version = app_version
        self.capabilities = dict(capabilities)
        self.admission_planner = admission_planner or AdmissionPlanner()
        self.worker_controller = worker_controller or WorkerController(instance_lease=instance_lease)
        self._clock = clock
        self._uuid_factory = uuid_factory or uuid.uuid4
        self._lock = threading.RLock()
        self._active: OwnedWorkerProcess | Any | None = None
        self._active_job_id: str | None = None
        self._pending_terminal: Mapping[str, Any] | None = None
        self._scheduling_paused = False
        if not self.database.read_only:
            self.database.install_component_schema(
                "scheduler", SCHEDULER_SCHEMA_VERSION, SCHEDULER_SCHEMA
            )
        self.events = EventLog(database)
        self.idempotency = IdempotencyStore(database, instance_id, clock=self._clock_value)

    def _clock_value(self) -> datetime:
        if self._clock is None:
            return datetime.now(timezone.utc)
        value = self._clock()
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str):
            return _parse_time(value)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), timezone.utc)
        raise TypeError("clock must return datetime, timestamp, or RFC 3339 text")

    def _now(self) -> str:
        return self._clock_value().isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _new_uuid(self) -> str:
        value = self._uuid_factory()
        return _canonical_uuid(str(value), "generated identifier")

    def _capability(self, operation: str) -> None:
        value = self.capabilities.get(operation)
        available = value if isinstance(value, bool) else value.get("available", False) if isinstance(value, Mapping) else False
        if not available:
            message = value.get("message") if isinstance(value, Mapping) else None
            raise ApiError(
                "CAPABILITY_UNAVAILABLE",
                message or f"The {operation} operation is not available in this runtime.",
            )

    @staticmethod
    def _progress(operation: str) -> dict[str, Any]:
        unit = "updates" if operation in {"tiny_train", "tiny_resume", "adapter_train", "adapter_resume"} else "records" if operation in {"evaluate", "retrieval_build", "retrieval_query"} else "tokens" if operation in {"generate", "chat_generate", "context_preview"} else "files"
        return {"current": 0, "total": None, "unit": unit, "message": ""}

    def _initial_job(
        self,
        *,
        job_id: str,
        operation: str,
        now: str,
        request_sha256: str,
        artifact_id: str,
        idempotency_key: str,
        queue_position: int,
    ) -> dict[str, Any]:
        return {
            "job_id": job_id,
            "operation": operation,
            "state": "queued",
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "progress": self._progress(operation),
            "result": None,
            "error": None,
            "last_cursor": 1,
            "phase": None,
            "step": None,
            "requested_final_step": None,
            "warnings": [],
            "warning_suppressed_count": 0,
            "checkpoint_boundary": None,
            "terminal_reason": None,
            "request": {"canonical_sha256": request_sha256, "artifact_id": artifact_id},
            "idempotency_key": idempotency_key,
            "run_id": None,
            "committed_artifact_count": 0,
            "log_artifact_ids": {"stdout": None, "stderr": None},
            "queue_position": queue_position,
            "phase_started_at": None,
            "hold_deadline_at": None,
        }

    def submit(
        self,
        request: Mapping[str, Any],
        idempotency_key: str,
        method: str = "POST",
        resolved_path: str = "/api/v1/jobs",
        origin: str = "locally_created",
    ) -> Submission:
        if not isinstance(request, Mapping):
            raise ApiError("VALIDATION_FAILED", reason_code="SCHEMA_INVALID")
        operation = request.get("operation")
        if operation not in OPERATIONS:
            raise ApiError("VALIDATION_FAILED", reason_code="SCHEMA_INVALID")
        schema_id = REQUEST_SCHEMAS[str(operation)]
        normalized_request = validate_schema(
            {"$ref": f"#/components/schemas/{schema_id}"},
            dict(request),
            document="openapi.json",
        )
        if not isinstance(normalized_request, dict):
            raise RuntimeError("job request schema did not normalize to an object")
        operation = normalized_request["operation"]
        canonical = canonical_json(normalized_request)
        request_sha256 = hashlib.sha256(canonical).hexdigest()
        replay = self.idempotency.lookup(
            method, resolved_path, idempotency_key, request_sha256
        )
        if replay is not None:
            return Submission(replay.status_code, replay.response_body, True)
        self.database.assert_writable()
        self._capability(str(operation))
        reservation = self.admission_planner.plan(normalized_request)
        with self._lock, self.database.transaction() as connection:
            replay = self.idempotency.lookup_in(
                connection, method, resolved_path, idempotency_key, request_sha256
            )
            if replay is not None:
                return Submission(replay.status_code, replay.response_body, True)
            queued = int(
                connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE state = 'queued'"
                ).fetchone()[0]
            )
            if queued >= MAX_QUEUED_JOBS:
                raise ApiError("QUEUE_FULL")
            job_id, now = self._new_uuid(), self._now()
            artifact_id = self.registry.accept_job_input(
                connection,
                job_id=job_id,
                canonical_bytes=canonical,
                canonical_sha256=request_sha256,
                schema_id=schema_id,
                origin=origin,
                reservation=reservation,
            )
            record = self._initial_job(
                job_id=job_id,
                operation=str(operation),
                now=now,
                request_sha256=request_sha256,
                artifact_id=artifact_id,
                idempotency_key=idempotency_key,
                queue_position=queued + 1,
            )
            connection.execute(
                """
                INSERT INTO jobs(
                    job_id, created_at, updated_at, state, operation, request_sha256,
                    idempotency_key, request_artifact_id, request_schema_id,
                    reservation_id, record_json
                ) VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    now,
                    now,
                    operation,
                    request_sha256,
                    idempotency_key,
                    artifact_id,
                    schema_id,
                    job_id,
                    canonical_json(record).decode("utf-8"),
                ),
            )
            self.events.append(
                connection,
                job_id=job_id,
                event_type="state_changed",
                payload={"state": "queued", "step": None, "requested_final_step": None},
                occurred_at=now,
            )
            body = self.idempotency.record_success(
                connection,
                method=method,
                resolved_path=resolved_path,
                key=idempotency_key,
                request_sha256=request_sha256,
                status_code=202,
                response_body=record,
            )
            self.database.bump_revision(connection)
        return Submission(202, body, False)

    @staticmethod
    def _decode_record(row: sqlite3.Row) -> dict[str, Any]:
        value = json.loads(row["record_json"])
        if not isinstance(value, dict):
            raise RuntimeError("stored job record is invalid")
        return value

    @staticmethod
    def _queue_position(connection: sqlite3.Connection, row: sqlite3.Row) -> int | None:
        if row["state"] != "queued":
            return None
        return int(
            connection.execute(
                """
                SELECT COUNT(*) FROM jobs
                WHERE state = 'queued'
                  AND (created_at < ? OR (created_at = ? AND job_id <= ?))
                """,
                (row["created_at"], row["created_at"], row["job_id"]),
            ).fetchone()[0]
        )

    def _public(self, connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
        record = self._decode_record(row)
        record["queue_position"] = self._queue_position(connection, row)
        return record

    def get(self, job_id: str) -> dict[str, Any]:
        _canonical_uuid(job_id, "job_id")
        with self.database.read() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise ApiError("NOT_FOUND")
            return self._public(connection, row)

    def list(
        self,
        state: str | None = None,
        operation: str | None = None,
        request_sha256: str | None = None,
        idempotency_key: str | None = None,
        cursor: str = "",
        limit: int = 50,
    ) -> dict[str, Any]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("job list limit must be between 1 and 100")
        filters = {
            "state": state,
            "operation": operation,
            "request_sha256": request_sha256,
            "idempotency_key": idempotency_key,
        }
        clauses: list[str] = []
        values: list[Any] = []
        for name, value in filters.items():
            if value is not None:
                clauses.append(f"{name} = ?")
                values.append(value)
        if cursor:
            created_at, item_id = self.registry.cursor_codec.decode(
                cursor, schema="jobs-v1", order="created_at-desc-id-desc", filters=filters
            )
            clauses.append("(created_at < ? OR (created_at = ? AND job_id < ?))")
            values.extend((created_at, created_at, item_id))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.database.read() as connection:
            rows = tuple(
                connection.execute(
                    "SELECT * FROM jobs" + where + " ORDER BY created_at DESC, job_id DESC LIMIT ?",
                    (*values, limit + 1),
                )
            )
            has_more = len(rows) > limit
            rows = rows[:limit]
            items = [self._public(connection, row) for row in rows]
        next_cursor = None
        if has_more and rows:
            last = rows[-1]
            next_cursor = self.registry.cursor_codec.encode(
                schema="jobs-v1",
                order="created_at-desc-id-desc",
                filters=filters,
                created_at=str(last["created_at"]),
                item_id=str(last["job_id"]),
            )
        return {"items": items, "next_cursor": next_cursor}

    def _save(self, connection: sqlite3.Connection, row: sqlite3.Row, record: Mapping[str, Any], **columns: Any) -> None:
        assignments = ["record_json = ?"]
        values: list[Any] = [canonical_json(dict(record)).decode("utf-8")]
        for name, value in columns.items():
            assignments.append(f"{name} = ?")
            values.append(value)
        values.append(row["job_id"])
        connection.execute(
            "UPDATE jobs SET " + ", ".join(assignments) + " WHERE job_id = ?", values
        )

    def _event(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        record: dict[str, Any],
        event_type: str,
        payload: Mapping[str, Any],
        now: str,
    ) -> None:
        event = self.events.append(
            connection,
            job_id=str(row["job_id"]),
            event_type=event_type,
            payload=payload,
            occurred_at=now,
        )
        record["last_cursor"] = event["cursor"]

    def _transition(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        record: dict[str, Any],
        state: str,
        now: str,
    ) -> None:
        record["state"] = state
        record["updated_at"] = now
        record["queue_position"] = None
        self._event(
            connection,
            row,
            record,
            "state_changed",
            {
                "state": state,
                "step": record["step"],
                "requested_final_step": record["requested_final_step"],
            },
            now,
        )

    def _terminal(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        record: dict[str, Any],
        *,
        state: str,
        reason: str,
        now: str,
        result: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
    ) -> None:
        self._transition(connection, row, record, state, now)
        record["finished_at"] = now
        record["phase"] = None
        record["phase_started_at"] = None
        record["hold_deadline_at"] = None
        record["terminal_reason"] = reason
        record["result"] = dict(result) if result is not None else None
        record["error"] = dict(error) if error is not None else None
        boundary = record["checkpoint_boundary"]
        self._event(
            connection,
            row,
            record,
            "terminal",
            {
                "state": state,
                "reason_code": reason,
                "checkpoint_id": boundary["checkpoint_id"] if boundary else None,
                "checkpoint_step": boundary["step"] if boundary else None,
            },
            now,
        )
        self.registry.release_capacity(connection, str(row["reservation_id"]))
        self._save(
            connection,
            row,
            record,
            state=state,
            updated_at=now,
            cancel_reason=row["cancel_reason"],
        )
        self.database.bump_revision(connection)

    def cancel(self, job_id: str, reason: str = "user_cancelled") -> dict[str, Any]:
        if reason != "user_cancelled":
            raise ValueError("public cancellation reason must be user_cancelled")
        with self._lock:
            worker = None
            with self.database.transaction() as connection:
                row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                if row is None:
                    raise ApiError("NOT_FOUND")
                record = self._public(connection, row)
                state, now = str(row["state"]), self._now()
                if state in {"interrupted", "cancelling"}:
                    return record
                if state in {"completed", "failed"}:
                    raise ApiError("STATE_CONFLICT")
                self._transition(connection, row, record, "cancelling", now)
                if state == "queued" or (state == "starting" and self._active_job_id != job_id):
                    self._terminal(
                        connection,
                        row,
                        record,
                        state="interrupted",
                        reason="user_cancelled",
                        now=now,
                    )
                else:
                    self._save(
                        connection,
                        row,
                        record,
                        state="cancelling",
                        updated_at=now,
                        cancel_requested_at=now,
                        cancel_reason=reason,
                    )
                    self.database.bump_revision(connection)
                    if self._active is not None and self._active_job_id == job_id:
                        worker = self._active
                updated = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                result = self._public(connection, updated)
            if worker is not None and not self._signal_cancel(worker, job_id, reason):
                return self.get(job_id)
            return result

    def list_events(self, job_id: str, after_cursor: int = 0, limit: int = 100) -> EventPage:
        return self.events.list(job_id, after_cursor, limit)

    def recover(self) -> int:
        """Resolve prior active states before sessions/control/scheduling start."""
        recovered = 0
        with self._lock:
            if self.database.read_only:
                self._scheduling_paused = True
                return 0
            try:
                with self.database.transaction() as connection:
                    rows = tuple(
                        connection.execute(
                            "SELECT * FROM jobs WHERE state IN ('starting','running','cancelling') ORDER BY created_at, job_id"
                        )
                    )
                    for row in rows:
                        record = self._decode_record(row)
                        self._terminal(
                            connection,
                            row,
                            record,
                            state="interrupted",
                            reason="service_restarted",
                            now=self._now(),
                        )
                        recovered += 1
                    for row in connection.execute("SELECT * FROM jobs"):
                        last = int(
                            connection.execute(
                                "SELECT COALESCE(MAX(cursor),0) FROM job_events WHERE job_id = ?",
                                (row["job_id"],),
                            ).fetchone()[0]
                        )
                        record = self._decode_record(row)
                        if last != int(record["last_cursor"]):
                            raise RuntimeError("job/event cursor mismatch")
            except (OSError, sqlite3.DatabaseError, RuntimeError, WorkerProtocolError):
                self.database.enter_read_only_recovery("STORAGE_CORRUPT")
                self._scheduling_paused = True
                return recovered
        return recovered

    def dispatch_once(self) -> dict[str, Any] | None:
        with self._lock:
            self.tick()
            if self._scheduling_paused or self.database.read_only or self._active is not None:
                return self.get(self._active_job_id) if self._active_job_id else None
            with self.database.transaction() as connection:
                row = connection.execute(
                    "SELECT * FROM jobs WHERE state = 'queued' ORDER BY created_at ASC, job_id ASC LIMIT 1"
                ).fetchone()
                if row is None:
                    return None
                record = self._decode_record(row)
                now = self._now()
                record["started_at"] = now
                self._transition(connection, row, record, "starting", now)
                self._save(connection, row, record, state="starting", updated_at=now)
                self.database.bump_revision(connection)
                job_id = str(row["job_id"])
                request_artifact_id = str(row["request_artifact_id"])
                request_sha256 = str(row["request_sha256"])
                schema_id = str(row["request_schema_id"])
            try:
                request_bytes = self.registry.read_artifact_bytes(request_artifact_id)
                if hashlib.sha256(request_bytes).hexdigest() != request_sha256:
                    raise WorkerProtocolError("request artifact digest changed")
                request = strict_json(request_bytes)
                if not isinstance(request, Mapping):
                    raise WorkerProtocolError("request artifact is not an object")
                validate_schema(
                    {"$ref": f"#/components/schemas/{schema_id}"},
                    request,
                    document="openapi.json",
                )
                worker = self.worker_controller.launch(
                    job_id=job_id,
                    instance_id=self.instance_id,
                    runtime_profile=self.runtime_profile,
                    root=self.root,
                    request=request,
                    request_sha256=request_sha256,
                    schema_id=schema_id,
                )
                self._active, self._active_job_id = worker, job_id
                self._pending_terminal = None
                with self.database.transaction() as connection:
                    live = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                    rec = self._decode_record(live)
                    self._save(
                        connection,
                        live,
                        rec,
                        spawn_nonce=worker.spawn_nonce,
                        worker_pid=worker.process.pid,
                    )
                return self.get(job_id)
            except BaseException:
                with self.database.transaction() as connection:
                    live = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                    rec = self._decode_record(live)
                    error = self._job_error(
                        "WORKER_PROTOCOL_ERROR", "The fixed worker could not be started."
                    )
                    self._terminal(
                        connection,
                        live,
                        rec,
                        state="failed",
                        reason=error["code"],
                        now=self._now(),
                        error=error,
                    )
                self._clear_active()
                return self.get(job_id)

    @staticmethod
    def _job_error(code: str, message: str) -> dict[str, Any]:
        row = ERROR_VOCABULARY.get(code, {})
        bindings = row.get("bindings") if isinstance(row, Mapping) else None
        binding = bindings.get("job", {}) if isinstance(bindings, Mapping) else row
        retryable = bool(binding.get("retryable", False)) if isinstance(binding, Mapping) else False
        return {"code": code, "message": message, "retryable": retryable, "field_errors": []}

    def _handle_worker_message(self, message: Mapping[str, Any]) -> None:
        job_id = self._active_job_id
        if job_id is None:
            raise WorkerProtocolError("worker message has no active job")
        kind = message["type"]
        with self.database.transaction() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise WorkerProtocolError("active job record is missing")
            record = self._decode_record(row)
            now = self._now()
            if kind == "ready":
                if row["state"] == "cancelling":
                    return
                if row["state"] != "starting":
                    raise WorkerProtocolError("ready acknowledgment in an invalid state")
                self._transition(connection, row, record, "running", now)
                self._save(connection, row, record, state="running", updated_at=now)
                self.database.bump_revision(connection)
                return
            if kind == "event":
                if row["state"] not in {"running", "cancelling"}:
                    raise WorkerProtocolError("worker event in an invalid state")
                event_type, payload = str(message["event_type"]), dict(message["payload"])
                if event_type == "phase_changed":
                    record["phase"] = payload["phase"]
                    record["phase_started_at"] = now
                    record["hold_deadline_at"] = (
                        (self._clock_value() + timedelta(seconds=600)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
                        if payload["phase"] == "cancellable_hold"
                        else None
                    )
                elif event_type == "progress":
                    record["progress"] = payload
                elif event_type == "warning":
                    if len(record["warnings"]) < 32:
                        record["warnings"].append(payload)
                    else:
                        record["warning_suppressed_count"] += 1
                        self._save(connection, row, record, updated_at=now)
                        return
                elif event_type == "checkpoint_committed":
                    record["checkpoint_boundary"] = {
                        "checkpoint_id": payload["checkpoint_id"],
                        "step": payload["step"],
                    }
                    record["step"] = payload["step"]
                self._event(connection, row, record, event_type, payload, now)
                record["updated_at"] = now
                self._save(connection, row, record, updated_at=now)
                return
            if kind == "result":
                if row["state"] == "cancelling":
                    raise WorkerProtocolError("worker completed after cancellation")
                self._terminal(
                    connection,
                    row,
                    record,
                    state="completed",
                    reason="completed",
                    now=now,
                    result=message["result"],
                )
            elif kind == "error":
                error = dict(message["error"])
                self._terminal(
                    connection,
                    row,
                    record,
                    state="failed",
                    reason=error["code"],
                    now=now,
                    error=error,
                )
            elif kind == "interrupted":
                self._terminal(
                    connection,
                    row,
                    record,
                    state="interrupted",
                    reason=str(message["reason_code"]),
                    now=now,
                    error=message["error"],
                )

    def _signal_cancel(self, worker: Any, job_id: str, reason: str) -> bool:
        try:
            worker.request_cancel(reason)
            return True
        except (OSError, WorkerProtocolError):
            try:
                worker.terminate_owned(worker.spawn_nonce)
            except (OSError, WorkerProtocolError):
                self._ownership_unknown(job_id)
            else:
                self._finish_interrupted(job_id, "worker_lost")
            self._clear_active()
            return False

    def _request_internal_cancel(self, reason: str) -> bool:
        if self._active_job_id is None or self._active is None:
            return False
        job_id, worker = self._active_job_id, self._active
        with self.database.transaction() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None or row["state"] in _TERMINAL:
                return False
            if row["state"] != "cancelling":
                record = self._decode_record(row)
                now = self._now()
                self._transition(connection, row, record, "cancelling", now)
                self._save(
                    connection,
                    row,
                    record,
                    state="cancelling",
                    updated_at=now,
                    cancel_requested_at=now,
                    cancel_reason=reason,
                )
                self.database.bump_revision(connection)
        return self._signal_cancel(worker, job_id, reason)

    def tick(self) -> dict[str, Any] | None:
        with self._lock:
            worker, job_id = self._active, self._active_job_id
            if worker is None or job_id is None:
                return None
            try:
                for message in worker.messages():
                    if message is None:
                        continue
                    if isinstance(message, BaseException):
                        raise WorkerProtocolError("worker protocol reader failed") from message
                    if message["type"] in {"result", "error", "interrupted"}:
                        if self._pending_terminal is not None:
                            raise WorkerProtocolError("worker emitted more than one terminal message")
                        self._pending_terminal = dict(message)
                    else:
                        if self._pending_terminal is not None:
                            raise WorkerProtocolError("worker emitted data after a terminal message")
                        self._handle_worker_message(message)
                job = self.get(job_id)
                now = self._clock_value()
                if job["started_at"] is not None and now >= _parse_time(job["started_at"]) + timedelta(seconds=DEADLINE_SECONDS) and job["state"] != "cancelling":
                    self._request_internal_cancel("timeout")
                    job = self.get(job_id)
                    if self._active is None:
                        return job
                if job["state"] == "cancelling":
                    with self.database.read() as connection:
                        row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                        requested = _parse_time(str(row["cancel_requested_at"]))
                        reason = str(row["cancel_reason"])
                    if now >= requested + timedelta(seconds=CANCEL_GRACE_SECONDS):
                        try:
                            worker.terminate_owned(worker.spawn_nonce)
                        except (OSError, WorkerProtocolError):
                            self._ownership_unknown(job_id)
                            return self.get(job_id)
                        self._finish_interrupted(job_id, reason)
                        self._clear_active()
                        return self.get(job_id)
                exit_code = worker.poll()
                protocol_eof = getattr(worker, "protocol_eof", True)
                if exit_code is not None and protocol_eof:
                    try:
                        worker.terminate_owned(worker.spawn_nonce)
                    except (OSError, WorkerProtocolError):
                        self._ownership_unknown(job_id)
                        self._clear_active()
                        return self.get(job_id)
                    pending = self._pending_terminal
                    if pending is not None:
                        if pending["type"] == "result" and exit_code != 0:
                            self._finish_failed(
                                job_id,
                                "WORKER_PROTOCOL_ERROR",
                                "The worker exited nonzero after a result message.",
                            )
                        else:
                            self._handle_worker_message(pending)
                    elif job["state"] == "cancelling":
                        self._finish_interrupted(job_id, "worker_lost")
                    else:
                        self._finish_failed(job_id, "WORKER_PROTOCOL_ERROR", "The worker exited without a terminal protocol message.")
                    self._clear_active()
                return self.get(job_id)
            except WorkerProtocolError:
                try:
                    worker.terminate_owned(worker.spawn_nonce)
                except (OSError, WorkerProtocolError):
                    self._ownership_unknown(job_id)
                else:
                    self._finish_failed(job_id, "WORKER_PROTOCOL_ERROR", "The worker emitted an invalid protocol message.")
                self._clear_active()
                return self.get(job_id)

    def _finish_interrupted(self, job_id: str, reason: str) -> None:
        with self.database.transaction() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None or row["state"] in _TERMINAL:
                return
            record = self._decode_record(row)
            self._terminal(
                connection,
                row,
                record,
                state="interrupted",
                reason=reason,
                now=self._now(),
            )

    def _finish_failed(self, job_id: str, code: str, message: str) -> None:
        with self.database.transaction() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None or row["state"] in _TERMINAL:
                return
            record = self._decode_record(row)
            error = self._job_error(code, message)
            self._terminal(
                connection,
                row,
                record,
                state="failed",
                reason=code,
                now=self._now(),
                error=error,
            )

    def _ownership_unknown(self, job_id: str) -> None:
        with self.database.transaction() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None or row["state"] in _TERMINAL:
                return
            record = self._decode_record(row)
            error = self._job_error(
                "WORKER_OWNERSHIP_UNKNOWN", "The worker ownership proof was lost."
            )
            self._terminal(
                connection,
                row,
                record,
                state="interrupted",
                reason="worker_ownership_unknown",
                now=self._now(),
                error=error,
            )
        self._scheduling_paused = True

    def _clear_active(self) -> None:
        if self._active is not None:
            try:
                self._active.close()
            except OSError:
                pass
        self._active = None
        self._active_job_id = None
        self._pending_terminal = None

    def shutdown(self) -> None:
        with self._lock:
            if self._active is None:
                return
            if self.database.read_only:
                worker = self._active
                worker.terminate_owned(worker.spawn_nonce)
                self._clear_active()
                return
            self._request_internal_cancel("service_shutdown")
        deadline = time.monotonic() + CANCEL_GRACE_SECONDS + 1.0
        while time.monotonic() < deadline:
            job = self.tick()
            if job is None or job["state"] in _TERMINAL:
                return
            time.sleep(0.05)
        self.tick()


__all__ = [
    "AdmissionPlanner",
    "IdempotencyCommit",
    "IdempotencyStore",
    "OwnedWorkerProcess",
    "Scheduler",
    "StoredResponse",
    "Submission",
    "WorkerController",
]
