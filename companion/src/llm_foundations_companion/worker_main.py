"""Fixed isolated worker entry point."""

from __future__ import annotations

import argparse
import ctypes
import os
import signal
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .errors import ERROR_VOCABULARY
from .operations import OperationFailure, OperationUnavailable, handler_for
from .worker_context import (
    CommitRejected,
    WorkerContext,
    WorkerInterrupted,
)
from .worker_protocol import (
    CancellationToken,
    WorkerProtocolError,
    open_inherited_stream,
    read_frame,
    validate_request_envelope,
    write_frame,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm_foundations_companion.worker_main")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--instance-id", required=True)
    return parser


def _retryable(code: str) -> bool:
    row = ERROR_VOCABULARY.get(code, {})
    bindings = row.get("bindings") if isinstance(row, Mapping) else None
    binding = bindings.get("job", {}) if isinstance(bindings, Mapping) else row
    return bool(binding.get("retryable", False)) if isinstance(binding, Mapping) else False


def _error(
    code: str,
    message: str,
    *,
    retryable: bool | None = None,
    field_errors: tuple[Mapping[str, str], ...] = (),
) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {
            "code": code,
            "message": message,
            "retryable": _retryable(code) if retryable is None else retryable,
            "field_errors": [dict(item) for item in field_errors],
        },
    }


def _interrupted(
    context: WorkerContext | None,
    reason_code: str,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    boundary = context.last_checkpoint if context is not None else None
    return {
        "type": "interrupted",
        "reason_code": reason_code,
        "checkpoint_id": boundary["checkpoint_id"] if boundary else None,
        "checkpoint_step": boundary["step"] if boundary else None,
        "error": dict(error) if error is not None else None,
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
    context: WorkerContext | None = None
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
                "input_snapshot_sha256": envelope["input_snapshot_sha256"],
            },
        )
        request = envelope["request"]
        context = WorkerContext(
            job_id=args.job_id,
            instance_id=args.instance_id,
            runtime_profile=profile,
            staging_path=staging,
            cancellation=cancellation,
            protocol=protocol_stream,
            control=request_stream,
            input_snapshot=envelope["input_snapshot"],
            output_allocations=envelope["output_allocations"],
        )
        if cancellation.cancelled:
            write_frame(
                protocol_stream,
                _interrupted(context, cancellation.reason or "user_cancelled"),
            )
            return 0
        result = handler_for(request["operation"])(request, context)
        if not isinstance(result, Mapping):
            raise WorkerProtocolError("operation result must be an object")
        if cancellation.cancelled:
            write_frame(
                protocol_stream,
                _interrupted(context, cancellation.reason or "user_cancelled"),
            )
            return 0
        write_frame(
            protocol_stream,
            {"type": "result", "operation": request["operation"], "result": dict(result)},
        )
        return 0
    except WorkerInterrupted as exc:
        if protocol_stream is not None:
            write_frame(
                protocol_stream,
                _interrupted(context, exc.reason_code, exc.error),
            )
        return 0
    except CommitRejected as exc:
        if protocol_stream is not None:
            write_frame(
                protocol_stream,
                {"type": "error", "error": dict(exc.error)},
            )
        return 2
    except OperationFailure as exc:
        if protocol_stream is not None:
            write_frame(
                protocol_stream,
                _error(
                    exc.code,
                    exc.message,
                    retryable=exc.retryable,
                    field_errors=exc.field_errors,
                ),
            )
        return 2
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
