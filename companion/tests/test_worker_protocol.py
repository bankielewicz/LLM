from __future__ import annotations

import hashlib
import io
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from llm_foundations_companion.operations import HANDLERS, OPERATIONS, OperationUnavailable
from llm_foundations_companion.scheduler import OwnedWorkerProcess, WorkerController
from llm_foundations_companion.schema import canonical_json
from llm_foundations_companion.worker_protocol import (
    PROTOCOL_VERSION,
    ProtocolState,
    WorkerProtocolError,
    make_request_envelope,
    read_frame,
    validate_event_payload,
    validate_request_envelope,
    write_frame,
)


JOB_ID = "123e4567-e89b-42d3-a456-426614174000"
INSTANCE_ID = "123e4567-e89b-42d3-a456-426614174001"
NONCE = "a" * 64
SCHEMA = "ModelPrepareRequest"
REQUEST = {
    "operation": "model_prepare",
    "model_profile_id": "smollm2-135m-instruct-v1",
    "accept_download": True,
}
DIGEST = hashlib.sha256(canonical_json(REQUEST)).hexdigest()


def test_framing_round_trips_one_canonical_object() -> None:
    stream = io.BytesIO()
    write_frame(stream, {"z": 1, "a": "é"})
    assert stream.getvalue()[4:] == canonical_json({"a": "é", "z": 1})
    stream.seek(0)
    assert read_frame(stream) == {"a": "é", "z": 1}
    assert read_frame(stream, allow_eof=True) is None


def test_request_envelope_binds_identity_schema_operation_and_digest() -> None:
    envelope = make_request_envelope(
        job_id=JOB_ID,
        instance_id=INSTANCE_ID,
        spawn_nonce=NONCE,
        schema_id=SCHEMA,
        request_sha256=DIGEST,
        request=REQUEST,
    )
    assert envelope["protocol_version"] == PROTOCOL_VERSION
    assert validate_request_envelope(
        envelope, job_id=JOB_ID, instance_id=INSTANCE_ID
    ) == envelope

    changed = dict(envelope)
    changed["request_sha256"] = "b" * 64
    with pytest.raises(WorkerProtocolError, match="digest"):
        validate_request_envelope(changed, job_id=JOB_ID, instance_id=INSTANCE_ID)

    changed = dict(envelope)
    changed["request"] = {"operation": "learner.module", "value": "uploaded.py"}
    with pytest.raises(WorkerProtocolError, match="closed dispatch"):
        validate_request_envelope(changed, job_id=JOB_ID, instance_id=INSTANCE_ID)


def test_protocol_state_requires_exact_ready_then_one_matching_terminal() -> None:
    state = ProtocolState(JOB_ID, INSTANCE_ID, NONCE, DIGEST, SCHEMA, "model_prepare")
    with pytest.raises(WorkerProtocolError, match="acknowledge"):
        state.accept({"type": "result", "operation": "model_prepare", "result": {}})
    ready = {
        "type": "ready",
        "protocol_version": PROTOCOL_VERSION,
        "job_id": JOB_ID,
        "instance_id": INSTANCE_ID,
        "spawn_nonce": NONCE,
        "request_sha256": DIGEST,
        "schema_id": SCHEMA,
    }
    assert state.accept(ready) == ready
    with pytest.raises(WorkerProtocolError, match="operation"):
        state.accept({"type": "result", "operation": "generate", "result": {}})

    state = ProtocolState(JOB_ID, INSTANCE_ID, NONCE, DIGEST, SCHEMA, "model_prepare")
    state.accept(ready)
    result = {"type": "result", "operation": "model_prepare", "result": {"ok": True}}
    assert state.accept(result) == result
    with pytest.raises(WorkerProtocolError, match="after a terminal"):
        state.accept(result)


def test_event_semantics_reject_nonfinite_and_incoherent_terminal_payloads() -> None:
    metric = {
        "phase": "validation",
        "step": 2,
        "split": "validation",
        "name": "validation_nll_token",
        "value": 1.25,
    }
    assert validate_event_payload("metric", metric) == metric
    with pytest.raises(WorkerProtocolError, match="finite"):
        validate_event_payload("metric", {**metric, "value": math.nan})
    with pytest.raises(WorkerProtocolError, match="wrong reason"):
        validate_event_payload(
            "terminal",
            {
                "state": "completed",
                "reason_code": "timeout",
                "checkpoint_id": None,
                "checkpoint_step": None,
            },
        )
    with pytest.raises(WorkerProtocolError, match="paired"):
        validate_event_payload(
            "terminal",
            {
                "state": "interrupted",
                "reason_code": "user_cancelled",
                "checkpoint_id": JOB_ID,
                "checkpoint_step": None,
            },
        )


def test_s1_production_dispatch_is_closed_and_defensively_unavailable() -> None:
    assert tuple(HANDLERS) == OPERATIONS
    assert len(HANDLERS) == 15
    for operation, handler in HANDLERS.items():
        with pytest.raises(OperationUnavailable) as caught:
            handler({"operation": operation}, object())
        assert caught.value.code == "CAPABILITY_UNAVAILABLE"
    assert WorkerController().command_prefix == (
        sys.executable,
        "-I",
        "-m",
        "llm_foundations_companion.worker_main",
    )


@pytest.mark.skipif(os.name == "nt", reason="WSL/POSIX process-group control")
def test_owned_group_force_stop_after_ignored_grace_spares_unrelated_process() -> None:
    worker_code = (
        "import signal,subprocess,sys,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        "time.sleep(60)"
    )
    worker = subprocess.Popen(
        [sys.executable, "-c", worker_code],
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    sentinel = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    request_read, request_write = os.pipe()
    cancel_read, cancel_write = os.pipe()
    protocol_read, protocol_write = os.pipe()
    nonce = "c" * 64
    owned = OwnedWorkerProcess(
        process=worker,
        job_id=JOB_ID,
        spawn_nonce=nonce,
        request_writer=os.fdopen(request_write, "wb", buffering=0),
        cancel_writer=os.fdopen(cancel_write, "wb", buffering=0),
        protocol_reader=os.fdopen(protocol_read, "rb", buffering=0),
        protocol_state=ProtocolState(
            JOB_ID, INSTANCE_ID, nonce, DIGEST, SCHEMA, "model_prepare"
        ),
        windows_job=None,
    )
    try:
        owned.request_cancel("user_cancelled")
        time.sleep(0.05)
        assert worker.poll() is None  # Cooperative grace does not scan or signal by name.
        with pytest.raises(WorkerProtocolError, match="nonce"):
            owned.terminate_owned("d" * 64)
        assert worker.poll() is None
        owned.terminate_owned(nonce)
        worker.wait(timeout=5)
        assert worker.returncode is not None
        assert sentinel.poll() is None
    finally:
        os.close(protocol_write)
        os.close(request_read)
        os.close(cancel_read)
        owned.close()
        if worker.poll() is None:
            os.killpg(worker.pid, 9)
            worker.wait(timeout=5)
        sentinel.terminate()
        sentinel.wait(timeout=5)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group descendant probe")
def test_owned_group_force_stop_kills_descendant_after_leader_exits(tmp_path) -> None:
    child_pid_path = tmp_path / "child.pid"
    leader_code = (
        "import pathlib,subprocess,sys;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid),encoding='ascii')"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", leader_code, str(child_pid_path)],
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    request_read, request_write = os.pipe()
    cancel_read, cancel_write = os.pipe()
    protocol_read, protocol_write = os.pipe()
    nonce = "e" * 64
    owned = OwnedWorkerProcess(
        process=leader,
        job_id=JOB_ID,
        spawn_nonce=nonce,
        request_writer=os.fdopen(request_write, "wb", buffering=0),
        cancel_writer=os.fdopen(cancel_write, "wb", buffering=0),
        protocol_reader=os.fdopen(protocol_read, "rb", buffering=0),
        protocol_state=ProtocolState(
            JOB_ID, INSTANCE_ID, nonce, DIGEST, SCHEMA, "model_prepare"
        ),
        windows_job=None,
    )
    child_pid = None
    try:
        leader.wait(timeout=5)
        deadline = time.monotonic() + 5
        while not child_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        child_pid = int(child_pid_path.read_text(encoding="ascii"))
        assert _process_is_running(child_pid)

        owned.terminate_owned(nonce)

        deadline = time.monotonic() + 5
        while _process_is_running(child_pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _process_is_running(child_pid)
    finally:
        os.close(protocol_write)
        os.close(request_read)
        os.close(cancel_read)
        owned.close()
        if child_pid is not None and _process_is_running(child_pid):
            os.kill(child_pid, 9)


def _process_is_running(pid: int) -> bool:
    status = Path(f"/proc/{pid}/stat")
    if not status.exists():
        return False
    try:
        return status.read_text(encoding="ascii").split()[2] != "Z"
    except (FileNotFoundError, IndexError):
        return False
