"""Fixed isolated worker entry point.

This module intentionally imports no model framework. Concrete operation
modules are a later slice; the S1 dispatch map fails closed.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import signal
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .operations import OperationUnavailable, handler_for
from .worker_protocol import (
    CancellationToken,
    WorkerProtocolError,
    open_inherited_stream,
    read_frame,
    validate_event_payload,
    validate_request_envelope,
    write_frame,
)


@dataclass(frozen=True)
class WorkerContext:
    job_id: str
    instance_id: str
    runtime_profile: str
    staging_path: Path
    cancellation: CancellationToken
    _protocol: Any

    def emit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        validate_event_payload(event_type, payload)
        if event_type in {"state_changed", "terminal"}:
            raise WorkerProtocolError("the scheduler owns state and terminal events")
        write_frame(
            self._protocol,
            {"type": "event", "event_type": event_type, "payload": dict(payload)},
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm_foundations_companion.worker_main")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--instance-id", required=True)
    return parser


def _error(code: str, message: str, *, retryable: bool = False) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "field_errors": [],
        },
    }


def _establish_parent_boundary() -> None:
    raw = os.environ.pop("LLMF_PARENT_PID", "")
    if not raw.isascii() or not raw.isdecimal() or int(raw) <= 0:
        raise WorkerProtocolError("worker parent identity is invalid")
    expected_parent = int(raw)
    if os.name != "nt":
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
        if os.getppid() != expected_parent:
            raise WorkerProtocolError("worker parent changed before request validation")


def _verify_environment_identity(job_id: str, instance_id: str) -> None:
    if os.environ.pop("LLMF_JOB_ID", "") != job_id:
        raise WorkerProtocolError("worker job environment identity is invalid")
    if os.environ.pop("LLMF_INSTANCE_ID", "") != instance_id:
        raise WorkerProtocolError("worker instance environment identity is invalid")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request_stream = protocol_stream = cancel_stream = None
    try:
        _establish_parent_boundary()
        _verify_environment_identity(args.job_id, args.instance_id)
        request_stream = open_inherited_stream("LLMF_REQUEST_FD", "rb")
        protocol_stream = open_inherited_stream("LLMF_PROTOCOL_FD", "wb")
        cancel_stream = open_inherited_stream("LLMF_CANCEL_FD", "rb")
        envelope = validate_request_envelope(
            read_frame(request_stream), job_id=args.job_id, instance_id=args.instance_id
        )
        expected_nonce = os.environ.pop("LLMF_SPAWN_NONCE", "")
        if expected_nonce != envelope["spawn_nonce"]:
            raise WorkerProtocolError("spawn nonce does not match the owned process")
        profile = os.environ.pop("LLMF_RUNTIME_PROFILE", "")
        if profile not in {"win-cpu", "win-cuda", "wsl-cpu", "wsl-cuda"}:
            raise WorkerProtocolError("runtime profile is invalid")
        staging = Path(os.environ.pop("LLMF_STAGING_PATH", ""))
        if not staging.is_absolute() or not staging.is_dir():
            raise WorkerProtocolError("worker staging directory is invalid")
        cancellation = CancellationToken(cancel_stream)
        write_frame(
            protocol_stream,
            {
                "type": "ready",
                "protocol_version": envelope["protocol_version"],
                "job_id": envelope["job_id"],
                "instance_id": envelope["instance_id"],
                "spawn_nonce": envelope["spawn_nonce"],
                "request_sha256": envelope["request_sha256"],
                "schema_id": envelope["schema_id"],
            },
        )
        if cancellation.cancelled:
            write_frame(
                protocol_stream,
                {
                    "type": "interrupted",
                    "reason_code": cancellation.reason or "user_cancelled",
                    "checkpoint_id": None,
                    "checkpoint_step": None,
                    "error": None,
                },
            )
            return 0
        request = envelope["request"]
        context = WorkerContext(
            job_id=args.job_id,
            instance_id=args.instance_id,
            runtime_profile=profile,
            staging_path=staging,
            cancellation=cancellation,
            _protocol=protocol_stream,
        )
        result = handler_for(request["operation"])(request, context)
        if not isinstance(result, Mapping):
            raise WorkerProtocolError("operation result must be an object")
        write_frame(
            protocol_stream,
            {"type": "result", "operation": request["operation"], "result": dict(result)},
        )
        return 0
    except OperationUnavailable:
        # Admission must prevent this path. If it occurs, fail the accepted job
        # with a legal asynchronous code and preserve the closed dispatch fact.
        if protocol_stream is not None:
            write_frame(
                protocol_stream,
                _error(
                    "WORKER_PROTOCOL_ERROR",
                    "The accepted operation has no installed worker implementation.",
                ),
            )
        return 2
    except (EOFError, OSError, WorkerProtocolError, ValueError):
        if protocol_stream is not None:
            try:
                write_frame(
                    protocol_stream,
                    _error("WORKER_PROTOCOL_ERROR", "The worker protocol was rejected."),
                )
            except (OSError, WorkerProtocolError):
                pass
        return 2
    except BaseException:
        if protocol_stream is not None:
            try:
                write_frame(
                    protocol_stream,
                    _error("INTERNAL_ERROR", "The worker could not complete the operation."),
                )
            except (OSError, WorkerProtocolError):
                pass
        return 3
    finally:
        for stream in (request_stream, protocol_stream, cancel_stream):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
