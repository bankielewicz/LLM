"""Worker entrypoint and context failure boundaries with explicit doubles."""

from __future__ import annotations

import copy
import hashlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

from llm_foundations_companion import worker_context, worker_main
from llm_foundations_companion.operations import OperationFailure, OperationUnavailable
from llm_foundations_companion.worker_context import CommitRejected, WorkerContext, WorkerInterrupted
from llm_foundations_companion.worker_protocol import CancellationToken, WorkerProtocolError, write_frame
from test_worker_protocol import ALLOCATIONS, INSTANCE_ID, JOB_ID, NONCE, SNAPSHOT


def _context(
    tmp_path: Path,
    *,
    snapshot=None,
    allocations=None,
    control=None,
    cancellation=None,
) -> WorkerContext:
    staging = tmp_path / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    snapshot = copy.deepcopy(SNAPSHOT if snapshot is None else snapshot)
    allocations = copy.deepcopy(ALLOCATIONS if allocations is None else allocations)
    return WorkerContext(
        job_id=JOB_ID,
        instance_id=INSTANCE_ID,
        runtime_profile="wsl-cpu",
        staging_path=staging,
        cancellation=cancellation or CancellationToken(io.BytesIO()),
        protocol=io.BytesIO(),
        control=control or io.BytesIO(),
        input_snapshot=snapshot,
        output_allocations=allocations,
    )


def _snapshot_with_input(staging: Path, *, path="inputs/data.bin", payload=b"fixed"):
    input_path = staging / path
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(payload)
    snapshot = copy.deepcopy(SNAPSHOT)
    snapshot["inputs"] = [{
        "role": "dataset_train",
        "artifact_id": JOB_ID,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "path": path,
    }]
    return snapshot, input_path


def test_context_rejects_allocation_and_snapshot_identity_mismatches(tmp_path: Path) -> None:
    allocations = copy.deepcopy(ALLOCATIONS)
    allocations["run_id"] = JOB_ID
    with pytest.raises(WorkerProtocolError, match="allocations"):
        _context(tmp_path, allocations=allocations)

    snapshot = copy.deepcopy(SNAPSHOT)
    snapshot["runtime_profile"] = "wsl-cuda"
    with pytest.raises(WorkerProtocolError, match="identity"):
        _context(tmp_path, snapshot=snapshot)


@pytest.mark.parametrize(
    ("parent", "message"),
    [
        (None, "no verified parent"),
        ({"checkpoint_id": "invalid", "step": 1}, "identity is invalid"),
        ({"checkpoint_id": JOB_ID.upper(), "step": 1}, "identity is invalid"),
        ({"checkpoint_id": JOB_ID, "step": True}, "step is invalid"),
    ],
)
def test_resume_context_rejects_invalid_parent_boundary(tmp_path: Path, parent, message) -> None:
    snapshot = copy.deepcopy(SNAPSHOT)
    snapshot["operation"] = "tiny_resume"
    snapshot["resolved"] = {} if parent is None else {"parent_checkpoint": parent}
    with pytest.raises(WorkerProtocolError, match=message):
        _context(tmp_path, snapshot=snapshot)


def test_context_rejects_scheduler_owned_events_and_missing_inputs(tmp_path: Path) -> None:
    context = _context(tmp_path)
    with pytest.raises(WorkerProtocolError, match="scheduler owns"):
        context.emit(
            "state_changed",
            {"state": "running", "step": 0, "requested_final_step": None},
        )
    with pytest.raises(WorkerProtocolError, match="role is unavailable"):
        context.input("dataset_train")


def test_context_caches_verified_input_and_detects_mutation(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    snapshot, path = _snapshot_with_input(staging)
    context = _context(tmp_path, snapshot=snapshot)
    first = context.input("dataset_train")
    assert context.input("dataset_train") is first

    changed = _context(tmp_path / "changed", snapshot=snapshot)
    changed_path = changed.staging_path / "inputs" / "data.bin"
    changed_path.parent.mkdir(parents=True)
    changed_path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(WorkerProtocolError, match="bytes changed"):
        changed.input("dataset_train")


def test_context_input_rejects_unconfined_link_and_non_file(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context._inputs["dataset_train"] = {"path": "outside/data.bin"}
    with pytest.raises(WorkerProtocolError, match="not confined"):
        context.input("dataset_train")

    inputs = context.staging_path / "inputs"
    inputs.mkdir()
    target = context.staging_path / "target.bin"
    target.write_bytes(b"x")
    link = inputs / "link.bin"
    link.symlink_to(target)
    context._inputs["dataset_train"] = {"path": "inputs/link.bin"}
    with pytest.raises(WorkerProtocolError, match="is a link"):
        context.input("dataset_train")

    directory = inputs / "directory"
    directory.mkdir()
    context._inputs["dataset_train"] = {"path": "inputs/directory"}
    with pytest.raises(WorkerProtocolError, match="not a regular file"):
        context.input("dataset_train")


def _control(message) -> io.BytesIO:
    stream = io.BytesIO()
    write_frame(stream, message)
    stream.seek(0)
    return stream


def test_parent_acknowledgments_bind_kind_and_rejection(tmp_path: Path) -> None:
    error = {
        "code": "CHECKPOINT_WRITE_FAILED",
        "message": "The staged artifact was rejected.",
        "retryable": False,
        "field_errors": [],
    }
    context = _context(
        tmp_path / "wrong-rejection",
        control=_control({"type": "commit_rejected", "kind": "checkpoint", "error": error}),
    )
    with pytest.raises(WorkerProtocolError, match="rejection kind"):
        context._read_ack("artifact")

    context = _context(
        tmp_path / "rejected",
        control=_control({"type": "commit_rejected", "kind": "artifact", "error": error}),
    )
    with pytest.raises(CommitRejected) as raised:
        context._read_ack("artifact")
    assert raised.value.kind == "artifact" and raised.value.error == error

    context = _context(
        tmp_path / "wrong-ack",
        control=_control({
            "type": "artifact_prepared",
            "role": "run_result",
            "artifact_id": INSTANCE_ID,
            "sha256": "a" * 64,
        }),
    )
    with pytest.raises(WorkerProtocolError, match="acknowledgment kind"):
        context._read_ack("checkpoint")


def test_stage_artifact_rejects_role_link_external_path_and_bad_ack(tmp_path: Path) -> None:
    context = _context(tmp_path / "base")
    with pytest.raises(WorkerProtocolError, match="closed vocabulary"):
        context.stage_artifact("unknown", b"x")

    target = context.staging_path / "target.bin"
    target.write_bytes(b"x")
    link = context.staging_path / "link.bin"
    link.symlink_to(target)
    with pytest.raises(WorkerProtocolError, match="is a link"):
        context.stage_artifact("run_result", link)

    external = tmp_path / "external.bin"
    external.write_bytes(b"x")
    with pytest.raises(WorkerProtocolError, match="outside"):
        context.stage_artifact("run_result", external)

    data = b"result"
    bad_ack = {
        "type": "artifact_prepared",
        "role": "run_result",
        "artifact_id": INSTANCE_ID,
        "sha256": "b" * 64,
    }
    context = _context(tmp_path / "ack", control=_control(bad_ack))
    with pytest.raises(WorkerProtocolError, match="does not match"):
        context.stage_artifact("run_result", data, filename="result.partial")


def test_stage_artifact_rejects_short_write(tmp_path: Path, monkeypatch) -> None:
    context = _context(tmp_path)

    class Writer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write(self, view):
            return 0

        def flush(self):
            pass

        def fileno(self):
            return 1

    monkeypatch.setattr(worker_context.os, "fdopen", lambda *a, **k: Writer())
    with pytest.raises(OSError, match="short staged artifact write"):
        context.stage_artifact("run_result", b"x", filename="short.partial")


@pytest.mark.parametrize("name", ["", ".", "..", "../escape", "nested/name", "windows\\name"])
def test_staging_names_must_be_portable_basenames(name) -> None:
    with pytest.raises(WorkerProtocolError, match="portable basename"):
        worker_context._portable_basename(name)


def test_directory_fsync_is_an_explicit_windows_noop(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(worker_context.os, "name", "nt")
    worker_context._fsync_directory(tmp_path)


_CHECKPOINT_NAMES = (
    "model.safetensors",
    "optimizer.safetensors",
    "rng.safetensors",
    "tokenizer.json",
    "config.json",
    "trainer_state.json",
    "manifest.json",
)


def _checkpoint_case(tmp_path: Path, *, acknowledgment=None, cancellation=None):
    checkpoint_id = str(uuid.uuid4())
    allocations = {
        "run_id": JOB_ID,
        "model_id": INSTANCE_ID,
        "tokenizer_id": None,
        "checkpoint_ids": [checkpoint_id],
    }
    snapshot = copy.deepcopy(SNAPSHOT)
    snapshot["ids"] = allocations
    staging = tmp_path / "staging"
    directory = staging / "checkpoint.partial"
    directory.mkdir(parents=True)
    files = []
    for name in _CHECKPOINT_NAMES:
        payload = f"fixed:{name}".encode()
        (directory / name).write_bytes(payload)
        files.append({
            "name": name,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    proposal = {
        "staging_name": directory.name,
        "manifest_sha256": next(row["sha256"] for row in files if row["name"] == "manifest.json"),
        "files": files,
    }
    if acknowledgment is None:
        control = io.BytesIO()
    else:
        if callable(acknowledgment):
            acknowledgment = acknowledgment(checkpoint_id, proposal)
        control = _control(acknowledgment)
    context = _context(
        tmp_path,
        snapshot=snapshot,
        allocations=allocations,
        control=control,
        cancellation=cancellation,
    )
    return context, directory, proposal, checkpoint_id


def _checkpoint_ack(checkpoint_id, proposal):
    return {
        "type": "checkpoint_committed",
        "checkpoint_id": checkpoint_id,
        "step": 1,
        "sha256": proposal["manifest_sha256"],
        "run_id": JOB_ID,
        "artifact_ids": {
            name: str(uuid.uuid5(uuid.NAMESPACE_URL, name)) for name in _CHECKPOINT_NAMES
        },
    }


def test_checkpoint_rejects_non_directory_and_undeclared_member(tmp_path: Path) -> None:
    context, candidate, proposal, _ = _checkpoint_case(tmp_path / "file")
    for child in candidate.iterdir():
        child.unlink()
    candidate.rmdir()
    candidate.write_bytes(b"not a directory")
    with pytest.raises(WorkerProtocolError, match="not a directory"):
        context.commit_checkpoint(proposal)

    context, directory, proposal, _ = _checkpoint_case(tmp_path / "extra")
    (directory / "undeclared.bin").write_bytes(b"extra")
    with pytest.raises(WorkerProtocolError, match="undeclared files"):
        context.commit_checkpoint(proposal)


def test_checkpoint_rejects_linked_and_changed_members(tmp_path: Path) -> None:
    context, directory, proposal, _ = _checkpoint_case(tmp_path / "link")
    member = directory / "model.safetensors"
    member.unlink()
    member.symlink_to(directory / "config.json")
    with pytest.raises(WorkerProtocolError, match="not a regular file"):
        context.commit_checkpoint(proposal)

    context, directory, proposal, _ = _checkpoint_case(tmp_path / "changed")
    (directory / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(WorkerProtocolError, match="bytes do not match"):
        context.commit_checkpoint(proposal)


def test_cancelled_checkpoint_rejection_becomes_interruption(tmp_path: Path) -> None:
    rejection = {
        "type": "commit_rejected",
        "kind": "checkpoint",
        "error": {
            "code": "CHECKPOINT_WRITE_FAILED",
            "message": "checkpoint rejected",
            "retryable": False,
            "field_errors": [],
        },
    }
    context, _, proposal, _ = _checkpoint_case(
        tmp_path,
        acknowledgment=rejection,
        cancellation=_Cancellation((True,), reason="service_shutdown"),
    )
    with pytest.raises(WorkerInterrupted) as raised:
        context.commit_checkpoint(proposal)
    assert raised.value.reason_code == "service_shutdown"
    assert raised.value.error["code"] == "CHECKPOINT_WRITE_FAILED"


def test_uncancelled_checkpoint_rejection_remains_commit_rejected(tmp_path: Path) -> None:
    rejection = {
        "type": "commit_rejected",
        "kind": "checkpoint",
        "error": {
            "code": "CHECKPOINT_WRITE_FAILED",
            "message": "checkpoint rejected",
            "retryable": False,
            "field_errors": [],
        },
    }
    context, _, proposal, _ = _checkpoint_case(
        tmp_path,
        acknowledgment=rejection,
        cancellation=_Cancellation((False,)),
    )
    with pytest.raises(CommitRejected):
        context.commit_checkpoint(proposal)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda ack: ack.update(checkpoint_id=INSTANCE_ID), "unleased identity"),
        (lambda ack: ack.update(run_id=INSTANCE_ID), "run identity"),
        (lambda ack: ack.update(sha256="f" * 64), "digest"),
    ],
)
def test_checkpoint_acknowledgment_binds_leased_identity_and_digest(tmp_path: Path, change, message) -> None:
    def changed_ack(checkpoint_id, proposal):
        acknowledgment = _checkpoint_ack(checkpoint_id, proposal)
        change(acknowledgment)
        return acknowledgment

    context, _, proposal, _ = _checkpoint_case(tmp_path, acknowledgment=changed_ack)
    with pytest.raises(WorkerProtocolError, match=message):
        context.commit_checkpoint(proposal)


class _Stream:
    def __init__(self, *, close_error=False) -> None:
        self.close_error = close_error
        self.closed = 0

    def close(self):
        self.closed += 1
        if self.close_error:
            raise OSError("synthetic close failure")


class _Cancellation:
    def __init__(self, states=(False,), reason="user_cancelled") -> None:
        self.states = iter(states)
        self.last = states[-1]
        self.reason = reason

    @property
    def cancelled(self):
        self.last = next(self.states, self.last)
        return self.last


class _MainContext:
    def __init__(self, **kwargs) -> None:
        self.last_checkpoint = {"checkpoint_id": JOB_ID, "step": 7}


def _run_main(
    monkeypatch,
    tmp_path: Path,
    handler,
    *,
    cancellation=(False, False),
    environment=None,
    fail_error_write=None,
    close_error=False,
):
    envelope = {
        "protocol_version": "1",
        "job_id": JOB_ID,
        "instance_id": INSTANCE_ID,
        "spawn_nonce": NONCE,
        "request_sha256": "a" * 64,
        "schema_id": "TinyTrainRequest",
        "input_snapshot_sha256": "b" * 64,
        "request": {"operation": "tiny_train"},
        "input_snapshot": {},
        "output_allocations": {},
    }
    streams = [_Stream(close_error=close_error) for _ in range(3)]
    written = []
    monkeypatch.setattr(worker_main, "_establish_parent_boundary", lambda: None)
    monkeypatch.setattr(worker_main, "_verify_environment_identity", lambda *a: None)
    monkeypatch.setattr(worker_main, "open_inherited_stream", lambda *a: streams.pop(0))
    monkeypatch.setattr(worker_main, "read_frame", lambda stream: envelope)
    monkeypatch.setattr(worker_main, "validate_request_envelope", lambda raw, **kwargs: raw)
    monkeypatch.setattr(worker_main, "CancellationToken", lambda stream: _Cancellation(cancellation))
    monkeypatch.setattr(worker_main, "WorkerContext", _MainContext)
    monkeypatch.setattr(worker_main, "handler_for", lambda operation: handler)

    def capture(stream, message):
        if fail_error_write is not None and message.get("type") == "error":
            raise fail_error_write("synthetic protocol write failure")
        written.append(message)

    monkeypatch.setattr(worker_main, "write_frame", capture)
    values = {
        "LLMF_SPAWN_NONCE": NONCE,
        "LLMF_RUNTIME_PROFILE": "wsl-cpu",
        "LLMF_STAGING_PATH": str(tmp_path),
    }
    values.update(environment or {})
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return worker_main.main(["--job-id", JOB_ID, "--instance-id", INSTANCE_ID]), written


def test_worker_error_builder_uses_job_binding_retryability(monkeypatch) -> None:
    monkeypatch.setattr(worker_main, "ERROR_VOCABULARY", {
        "BOUND": {"bindings": {"job": {"retryable": True}}},
        "FLAT": {"retryable": False},
        "INVALID": "closed",
    })
    assert worker_main._error("BOUND", "message")["error"]["retryable"] is True
    assert worker_main._error("FLAT", "message")["error"]["retryable"] is False
    assert worker_main._error("INVALID", "message")["error"]["retryable"] is False


@pytest.mark.parametrize("raw", ["", "0", "-1", "１２"])
def test_parent_boundary_rejects_invalid_identity(monkeypatch, raw) -> None:
    monkeypatch.setenv("LLMF_PARENT_PID", raw)
    with pytest.raises(WorkerProtocolError, match="parent identity"):
        worker_main._establish_parent_boundary()


def test_parent_boundary_rejects_prctl_failure_and_parent_change(monkeypatch) -> None:
    fake = SimpleNamespace(prctl=lambda *a: 1)
    monkeypatch.setenv("LLMF_PARENT_PID", "123")
    monkeypatch.setattr(worker_main.ctypes, "CDLL", lambda *a, **k: fake)
    with pytest.raises(OSError, match="prctl"):
        worker_main._establish_parent_boundary()

    fake.prctl = lambda *a: 0
    monkeypatch.setenv("LLMF_PARENT_PID", "123")
    monkeypatch.setattr(worker_main.os, "getppid", lambda: 456)
    with pytest.raises(WorkerProtocolError, match="parent changed"):
        worker_main._establish_parent_boundary()


def test_environment_identity_consumes_exact_job_and_instance(monkeypatch) -> None:
    monkeypatch.setenv("LLMF_JOB_ID", "wrong")
    monkeypatch.setenv("LLMF_INSTANCE_ID", INSTANCE_ID)
    with pytest.raises(WorkerProtocolError, match="job environment"):
        worker_main._verify_environment_identity(JOB_ID, INSTANCE_ID)
    monkeypatch.setenv("LLMF_JOB_ID", JOB_ID)
    monkeypatch.setenv("LLMF_INSTANCE_ID", "wrong")
    with pytest.raises(WorkerProtocolError, match="instance environment"):
        worker_main._verify_environment_identity(JOB_ID, INSTANCE_ID)


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"LLMF_SPAWN_NONCE": "wrong"}, "spawn nonce"),
        ({"LLMF_RUNTIME_PROFILE": "unknown"}, "runtime profile"),
        ({"LLMF_STAGING_PATH": "relative"}, "staging directory"),
    ],
)
def test_worker_main_rejects_owned_environment_mismatch(monkeypatch, tmp_path, environment, message) -> None:
    code, written = _run_main(monkeypatch, tmp_path, lambda *a: {}, environment=environment)
    assert code == 2 and written[-1]["error"]["code"] == "WORKER_PROTOCOL_ERROR"


def test_worker_main_interrupts_before_and_after_handler(monkeypatch, tmp_path) -> None:
    called = []
    code, written = _run_main(
        monkeypatch, tmp_path, lambda *a: called.append(True) or {}, cancellation=(True,)
    )
    assert code == 0 and not called and written[-1]["type"] == "interrupted"
    code, written = _run_main(
        monkeypatch, tmp_path, lambda *a: {"ok": True}, cancellation=(False, True)
    )
    assert code == 0 and written[-1]["type"] == "interrupted"


@pytest.mark.parametrize(
    ("failure", "exit_code", "terminal_type", "error_code"),
    [
        (WorkerInterrupted("timeout"), 0, "interrupted", None),
        (CommitRejected("artifact", {"code": "ARTIFACT_WRITE_FAILED", "message": "rejected", "retryable": False, "field_errors": []}), 2, "error", "ARTIFACT_WRITE_FAILED"),
        (OperationFailure("DATASET_INVALID", "bad data", retryable=False), 2, "error", "DATASET_INVALID"),
        (OperationUnavailable("closed"), 2, "error", "WORKER_PROTOCOL_ERROR"),
        (WorkerProtocolError("bad protocol"), 2, "error", "WORKER_PROTOCOL_ERROR"),
        (RuntimeError("private failure"), 3, "error", "INTERNAL_ERROR"),
    ],
)
def test_worker_main_maps_handler_failures_to_closed_terminal(monkeypatch, tmp_path, failure, exit_code, terminal_type, error_code) -> None:
    def handler(*args):
        raise failure

    code, written = _run_main(monkeypatch, tmp_path, handler, close_error=True)
    assert code == exit_code and written[-1]["type"] == terminal_type
    if error_code is not None:
        assert written[-1]["error"]["code"] == error_code


def test_worker_main_rejects_non_object_result(monkeypatch, tmp_path) -> None:
    code, written = _run_main(monkeypatch, tmp_path, lambda *a: [])
    assert code == 2 and written[-1]["error"]["code"] == "WORKER_PROTOCOL_ERROR"


@pytest.mark.parametrize(
    ("failure", "write_failure", "exit_code"),
    [(WorkerProtocolError("bad"), WorkerProtocolError, 2), (RuntimeError("bad"), OSError, 3)],
)
def test_worker_main_contains_terminal_write_failure(monkeypatch, tmp_path, failure, write_failure, exit_code) -> None:
    def handler(*args):
        raise failure

    code, written = _run_main(monkeypatch, tmp_path, handler, fail_error_write=write_failure)
    assert code == exit_code and written[0]["type"] == "ready"
