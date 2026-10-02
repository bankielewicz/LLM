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

from llm_foundations_companion.database import Database
from llm_foundations_companion.operations import HANDLERS, OPERATIONS, OperationUnavailable
from llm_foundations_companion.preflight import S2_OPERATION_NAMES
from llm_foundations_companion.registry import Registry
from llm_foundations_companion.scheduler import OwnedWorkerProcess, WorkerController
from llm_foundations_companion.schema import canonical_json
from llm_foundations_companion.training_store import AdmissionPlan, TrainingStore
from llm_foundations_companion.worker_context import WorkerContext
from llm_foundations_companion.worker_protocol import (
    CancellationToken,
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
ALLOCATIONS = {
    "run_id": None,
    "model_id": None,
    "tokenizer_id": None,
    "checkpoint_ids": [],
}
SNAPSHOT = {
    "format": "llm-foundations-worker-input-v1",
    "job_id": JOB_ID,
    "operation": "model_prepare",
    "request_sha256": DIGEST,
    "runtime_profile": "wsl-cpu",
    "device": "cpu",
    "dependency_lock_sha256": "b" * 64,
    "companion_source_revision": "c" * 40,
    "ids": ALLOCATIONS,
    "resolved": {},
    "inputs": [],
}
SNAPSHOT_DIGEST = hashlib.sha256(canonical_json(SNAPSHOT)).hexdigest()


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
        input_snapshot_sha256=SNAPSHOT_DIGEST,
        input_snapshot=SNAPSHOT,
        output_allocations=ALLOCATIONS,
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
    state = ProtocolState(
        JOB_ID,
        INSTANCE_ID,
        NONCE,
        DIGEST,
        SCHEMA,
        "model_prepare",
        SNAPSHOT_DIGEST,
    )
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
        "input_snapshot_sha256": SNAPSHOT_DIGEST,
    }
    assert state.accept(ready) == ready
    with pytest.raises(WorkerProtocolError, match="operation"):
        state.accept({"type": "result", "operation": "generate", "result": {}})

    state = ProtocolState(
        JOB_ID,
        INSTANCE_ID,
        NONCE,
        DIGEST,
        SCHEMA,
        "model_prepare",
        SNAPSHOT_DIGEST,
    )
    state.accept(ready)
    result = {"type": "result", "operation": "model_prepare", "result": {"ok": True}}
    assert state.accept(result) == result
    with pytest.raises(WorkerProtocolError, match="after a terminal"):
        state.accept(result)


def test_protocol_blocks_after_staged_output_until_matching_parent_ack() -> None:
    state = ProtocolState(
        JOB_ID,
        INSTANCE_ID,
        NONCE,
        DIGEST,
        SCHEMA,
        "model_prepare",
        SNAPSHOT_DIGEST,
    )
    ready = {
        "type": "ready",
        "protocol_version": PROTOCOL_VERSION,
        "job_id": JOB_ID,
        "instance_id": INSTANCE_ID,
        "spawn_nonce": NONCE,
        "request_sha256": DIGEST,
        "schema_id": SCHEMA,
        "input_snapshot_sha256": SNAPSHOT_DIGEST,
    }
    state.accept(ready)
    proposal = {
        "type": "artifact_ready",
        "role": "run_result",
        "staging_name": "result.partial",
        "size_bytes": 2,
        "sha256": hashlib.sha256(b"{}").hexdigest(),
    }
    state.accept(proposal)
    with pytest.raises(WorkerProtocolError, match="before parent"):
        state.accept(
            {
                "type": "event",
                "event_type": "phase_changed",
                "payload": {"phase": "loading"},
            }
        )
    state.acknowledge(
        {
            "type": "artifact_prepared",
            "role": "run_result",
            "artifact_id": INSTANCE_ID,
            "sha256": proposal["sha256"],
        }
    )
    state.accept(
        {
            "type": "event",
            "event_type": "phase_changed",
            "payload": {"phase": "loading"},
        }
    )


def test_worker_context_verifies_snapshot_input_and_waits_for_artifact_prepare(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    inputs = staging / "inputs"
    inputs.mkdir(parents=True)
    source = inputs / "train.jsonl"
    source.write_bytes(b'{"record_id":"r1","text":"x"}\n')
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    allocations = {
        "run_id": JOB_ID,
        "model_id": INSTANCE_ID,
        "tokenizer_id": None,
        "checkpoint_ids": [],
    }
    snapshot = {
        **SNAPSHOT,
        "ids": allocations,
        "inputs": [
            {
                "role": "dataset_train",
                "artifact_id": JOB_ID,
                "sha256": source_digest,
                "size_bytes": source.stat().st_size,
                "path": "inputs/train.jsonl",
            }
        ],
    }
    output = b'{"ok":true}'
    output_digest = hashlib.sha256(output).hexdigest()
    retained = staging / "retained.worker"
    retained.write_bytes(b"keep")
    control = io.BytesIO()
    write_frame(
        control,
        {
            "type": "artifact_prepared",
            "role": "run_result",
            "artifact_id": INSTANCE_ID,
            "sha256": output_digest,
        },
    )
    control.seek(0)
    protocol = io.BytesIO()
    context = WorkerContext(
        job_id=JOB_ID,
        instance_id=INSTANCE_ID,
        runtime_profile="wsl-cpu",
        staging_path=staging,
        cancellation=CancellationToken(io.BytesIO()),
        protocol=protocol,
        control=control,
        input_snapshot=snapshot,
        output_allocations=allocations,
    )
    assert context.input("dataset_train").sha256 == source_digest
    acknowledgment = context.stage_artifact(
        "run_result", output, filename="result.partial"
    )
    assert acknowledgment["artifact_id"] == INSTANCE_ID
    assert not (staging / "result.partial").exists()
    assert retained.read_bytes() == b"keep"
    protocol.seek(0)
    assert read_frame(protocol)["type"] == "artifact_ready"


def test_worker_context_verifies_and_commits_exact_checkpoint_directory(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    checkpoint = staging / "checkpoint.partial"
    checkpoint.mkdir(parents=True)
    retained = staging / "retained.worker"
    retained.write_bytes(b"keep")
    names = (
        "model.safetensors",
        "optimizer.safetensors",
        "rng.safetensors",
        "tokenizer.json",
        "config.json",
        "trainer_state.json",
        "manifest.json",
    )
    files = []
    for name in names:
        payload = f"fixed:{name}".encode("utf-8")
        (checkpoint / name).write_bytes(payload)
        files.append(
            {
                "name": name,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest_sha256 = next(
        item["sha256"] for item in files if item["name"] == "manifest.json"
    )
    checkpoint_id = str(uuid.uuid4())
    allocations = {
        "run_id": JOB_ID,
        "model_id": INSTANCE_ID,
        "tokenizer_id": None,
        "checkpoint_ids": [checkpoint_id],
    }
    snapshot = {**SNAPSHOT, "ids": allocations}
    acknowledgment = {
        "type": "checkpoint_committed",
        "checkpoint_id": checkpoint_id,
        "step": 25,
        "sha256": manifest_sha256,
        "run_id": JOB_ID,
        "artifact_ids": {
            name: str(uuid.uuid5(uuid.NAMESPACE_URL, name)) for name in names
        },
    }
    control = io.BytesIO()
    write_frame(control, acknowledgment)
    control.seek(0)
    protocol = io.BytesIO()
    context = WorkerContext(
        job_id=JOB_ID,
        instance_id=INSTANCE_ID,
        runtime_profile="wsl-cpu",
        staging_path=staging,
        cancellation=CancellationToken(io.BytesIO()),
        protocol=protocol,
        control=control,
        input_snapshot=snapshot,
        output_allocations=allocations,
    )

    committed = context.commit_checkpoint(
        {
            "staging_name": checkpoint.name,
            "manifest_sha256": manifest_sha256,
            "files": files,
        }
    )

    assert committed == acknowledgment
    assert context.last_checkpoint == acknowledgment
    assert not checkpoint.exists()
    assert retained.read_bytes() == b"keep"
    protocol.seek(0)
    proposal = read_frame(protocol)
    assert proposal["type"] == "checkpoint_ready"
    assert {item["name"] for item in proposal["files"]} == set(names)
    with pytest.raises(WorkerProtocolError, match="exhausted"):
        context.checkpoint_identity()


def test_resume_worker_context_starts_at_verified_parent_boundary(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    parent_id = str(uuid.uuid4())
    allocations = {
        "run_id": JOB_ID,
        "model_id": INSTANCE_ID,
        "tokenizer_id": None,
        "checkpoint_ids": [str(uuid.uuid4())],
    }
    snapshot = {
        **SNAPSHOT,
        "operation": "tiny_resume",
        "ids": allocations,
        "resolved": {
            "parent_checkpoint": {
                "checkpoint_id": parent_id,
                "step": 25,
            }
        },
    }

    context = WorkerContext(
        job_id=JOB_ID,
        instance_id=INSTANCE_ID,
        runtime_profile="wsl-cpu",
        staging_path=staging,
        cancellation=CancellationToken(io.BytesIO()),
        protocol=io.BytesIO(),
        control=io.BytesIO(),
        input_snapshot=snapshot,
        output_allocations=allocations,
    )

    assert context.last_checkpoint == {"checkpoint_id": parent_id, "step": 25}


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



class _SpawnBoundary(RuntimeError):
    pass


def _materialized_worker_snapshot(
    tmp_path: Path,
) -> tuple[Path, dict[str, object], object]:
    root = tmp_path / "root"
    database = Database(root)
    database.initialize()
    registry = Registry(database, root, str(uuid.uuid4()), os.urandom(32))
    training = TrainingStore(
        database,
        registry,
        runtime_profile="wsl-cpu",
        device="cpu",
        dependency_lock_sha256="1" * 64,
        companion_source_revision="2" * 40,
    )
    input_bytes = b'{"record_id":"r1","text":"verified"}\n'
    artifact = registry.register_stream(
        io.BytesIO(input_bytes),
        str(uuid.uuid4()),
        artifact_type="dataset_split",
        display_name="train.jsonl",
        media_type="application/x-ndjson",
        preview_policy="text",
        origin="locally_created",
    )
    request: dict[str, object] = {"operation": "context_preview"}
    request_sha256 = hashlib.sha256(canonical_json(request)).hexdigest()
    job_id = str(uuid.uuid4())
    plan = AdmissionPlan(
        operation="context_preview",
        reservation={
            "byte_count": len(input_bytes),
            "artifact_rows": 1,
            "dataset_rows": 0,
            "run_rows": 0,
            "model_rows": 0,
            "checkpoint_rows": 0,
        },
        resolved={
            "request": request,
            "_inputs": [
                {
                    "role": "dataset.train",
                    "artifact_id": artifact["artifact_id"],
                    "sha256": artifact["sha256"],
                    "size_bytes": artifact["size_bytes"],
                    "sealed": False,
                }
            ],
        },
        run_id=None,
        model_id=None,
        tokenizer_id=None,
        checkpoint_ids=(),
    )
    with database.transaction() as connection:
        training.create_job_context_in(
            connection,
            job_id=job_id,
            request_sha256=request_sha256,
            plan=plan,
        )
    return root, request, training.materialize_worker_snapshot(job_id)


def _launch_materialized_snapshot(
    root: Path,
    request: dict[str, object],
    snapshot: object,
) -> None:
    WorkerController().launch(
        job_id=snapshot.value["job_id"],
        instance_id=INSTANCE_ID,
        runtime_profile="wsl-cpu",
        root=root,
        request=request,
        request_sha256=snapshot.value["request_sha256"],
        schema_id="ContextPreviewRequest",
        input_snapshot=snapshot.value,
        input_snapshot_sha256=snapshot.sha256,
        output_allocations=snapshot.value["ids"],
    )


def test_worker_launch_accepts_real_materialized_input_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, request, snapshot = _materialized_worker_snapshot(tmp_path)

    def spawn_boundary(*_args: object, **_kwargs: object) -> None:
        raise _SpawnBoundary

    monkeypatch.setattr(subprocess, "Popen", spawn_boundary)
    with pytest.raises(_SpawnBoundary):
        _launch_materialized_snapshot(root, request, snapshot)


def test_worker_launch_rejects_undeclared_staging_entry_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, request, snapshot = _materialized_worker_snapshot(tmp_path)
    staging = root / "jobs" / snapshot.value["job_id"] / "staging"
    (staging / "undeclared.bin").write_bytes(b"undeclared")
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("worker spawn was reached"),
    )

    with pytest.raises(OSError, match="undeclared"):
        _launch_materialized_snapshot(root, request, snapshot)


def test_worker_launch_rejects_symlinked_materialized_input_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, request, snapshot = _materialized_worker_snapshot(tmp_path)
    staging = root / "jobs" / snapshot.value["job_id"] / "staging"
    input_path = staging / snapshot.value["inputs"][0]["path"]
    outside = tmp_path / "outside-input"
    outside.write_bytes(input_path.read_bytes())
    input_path.unlink()
    try:
        input_path.symlink_to(outside)
    except OSError:
        pytest.skip("host does not permit symlink creation")
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("worker spawn was reached"),
    )

    with pytest.raises(OSError, match="confined regular file"):
        _launch_materialized_snapshot(root, request, snapshot)


def test_worker_launch_rejects_same_size_input_tamper_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, request, snapshot = _materialized_worker_snapshot(tmp_path)
    staging = root / "jobs" / snapshot.value["job_id"] / "staging"
    input_path = staging / snapshot.value["inputs"][0]["path"]
    original = input_path.read_bytes()
    input_path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("worker spawn was reached"),
    )

    with pytest.raises(OSError, match="bytes differ"):
        _launch_materialized_snapshot(root, request, snapshot)

def test_s2_production_dispatch_is_closed_and_keeps_later_slices_unavailable() -> None:
    assert tuple(HANDLERS) == OPERATIONS
    assert len(HANDLERS) == 15
    implemented = {
        "tokenizer_train",
        "tiny_train",
        "tiny_resume",
        "evaluate",
        "generate",
        "context_preview",
    }
    assert implemented == S2_OPERATION_NAMES
    for operation, handler in HANDLERS.items():
        if operation in implemented:
            assert handler.__module__ == "llm_foundations_companion.operations"
            continue
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
            JOB_ID, INSTANCE_ID, nonce, DIGEST, SCHEMA, "model_prepare",
            SNAPSHOT_DIGEST,
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
            JOB_ID, INSTANCE_ID, nonce, DIGEST, SCHEMA, "model_prepare",
            SNAPSHOT_DIGEST,
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
