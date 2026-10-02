"""Scheduler ownership, admission, replay, and cleanup boundary coverage."""

from __future__ import annotations

import copy
import io
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import uuid

import pytest

from llm_foundations_companion import scheduler as scheduler_module
from llm_foundations_companion.database import Database
from llm_foundations_companion.errors import ApiError
from llm_foundations_companion.registry import Registry
from llm_foundations_companion.scheduler import (
    AdmissionPlanner,
    IdempotencyStore,
    OwnedWorkerProcess,
    Scheduler,
    Submission,
    WorkerController,
)
from llm_foundations_companion.schema import canonical_json
from llm_foundations_companion.worker_protocol import WorkerProtocolError
from test_scheduler import (
    Clock,
    INSTALLATION_ID,
    INSTANCE_ID,
    ScriptedController,
    ScriptedOperationStore,
    ScriptedWorker,
    TinyReservation,
    model_prepare,
    set_requested_final_step,
    tiny_train,
)
from test_worker_protocol import ALLOCATIONS, SNAPSHOT


@pytest.mark.parametrize("value", [None, "invalid", INSTANCE_ID.upper()])
def test_scheduler_uuid_parser_rejects_noncanonical_values(value) -> None:
    with pytest.raises(ValueError, match="lowercase UUID"):
        scheduler_module._canonical_uuid(value, "value")


def test_scheduler_byte_and_submission_boundaries() -> None:
    assert scheduler_module._as_bytes(memoryview(b"fixed")) == b"fixed"
    assert scheduler_module._as_bytes(bytearray(b"fixed")) == b"fixed"
    with pytest.raises(TypeError, match="response body"):
        scheduler_module._as_bytes("fixed")
    with pytest.raises(RuntimeError, match="not an object"):
        _ = Submission(200, b"[]", False).job
    assert scheduler_module._parse_time(scheduler_module._utc_now()).tzinfo is not None


@pytest.mark.parametrize(
    ("clock", "aware"),
    [
        (lambda: "2026-10-01T12:00:00Z", True),
        (lambda: scheduler_module.datetime(2026, 10, 1, 12, 0), True),
    ],
)
def test_idempotency_clock_accepts_rfc3339_and_naive_datetime(clock, aware) -> None:
    store = IdempotencyStore(SimpleNamespace(), INSTANCE_ID, clock=clock)
    assert (store._now().tzinfo is not None) is aware


def test_idempotency_clock_rejects_unknown_type() -> None:
    store = IdempotencyStore(SimpleNamespace(), INSTANCE_ID, clock=lambda: object())
    with pytest.raises(TypeError, match="clock must return"):
        store._now()


@pytest.mark.parametrize(
    ("method", "path", "digest", "message"),
    [
        ("GET", "/api/v1/jobs", "a" * 64, "mutation"),
        ("POST", "/other", "a" * 64, "resolved API path"),
        ("POST", "/api/v1/jobs", "a" * 63, "digest"),
        ("POST", "/api/v1/jobs", "g" * 64, "digest"),
        ("POST", "/api/v1/jobs", "A" * 64, "digest"),
    ],
)
def test_idempotency_scope_rejects_unbound_route_and_digest(method, path, digest, message) -> None:
    with pytest.raises(ValueError, match=message):
        IdempotencyStore._scope(method, path, INSTANCE_ID, digest)


def test_idempotency_success_cannot_change_inside_retention(tmp_path: Path) -> None:
    database = Database(tmp_path / "database")
    database.initialize()
    database.install_component_schema(
        "scheduler",
        scheduler_module.SCHEDULER_SCHEMA_VERSION,
        scheduler_module.SCHEDULER_SCHEMA,
    )
    store = IdempotencyStore(
        database,
        INSTANCE_ID,
        clock=lambda: "2026-10-01T12:00:00Z",
    )
    values = dict(
        method="POST",
        resolved_path="/api/v1/jobs",
        key=str(uuid.uuid4()),
        request_sha256="a" * 64,
        status_code=202,
        response_body={"ok": True},
    )
    with database.transaction() as connection:
        store.record_success(connection, **values)
    with database.transaction() as connection:
        with pytest.raises(RuntimeError, match="success changed"):
            store.record_success(connection, **{**values, "status_code": 201})
    with database.transaction() as connection:
        with pytest.raises(ApiError) as raised:
            store.record_success(
                connection,
                **{**values, "request_sha256": "b" * 64},
            )
    assert raised.value.code == "IDEMPOTENCY_CONFLICT"
    database.close()


@pytest.mark.parametrize(
    ("operation", "byte_count"),
    [
        ("tokenizer_train", 64 * 1024 * 1024),
        ("model_prepare", 272_437_573 + 512 * 1024 * 1024),
        ("evaluate", 256 * 1024 * 1024),
        ("validate_bundle", 64 * 1024 * 1024),
    ],
)
def test_admission_planner_uses_operation_specific_closed_estimates(operation, byte_count) -> None:
    assert AdmissionPlanner().plan({"operation": operation})["byte_count"] == byte_count


@pytest.mark.parametrize("operation", ["tiny_train", "unknown"])
def test_admission_planner_rejects_uninstalled_and_unknown_operations(operation) -> None:
    with pytest.raises(ApiError) as raised:
        AdmissionPlanner().plan({"operation": operation})
    assert raised.value.code == "CAPABILITY_UNAVAILABLE"


def test_admission_planner_rejects_estimate_above_root_quota(monkeypatch) -> None:
    monkeypatch.setattr(scheduler_module, "ROOT_QUOTA_BYTES", 1)
    with pytest.raises(ApiError) as raised:
        AdmissionPlanner().plan({"operation": "tokenizer_train"})
    assert raised.value.reason_code == "ESTIMATE_EXCEEDS_ROOT_QUOTA"


class _ChunkStream:
    def __init__(self, chunks, *, close_error=False) -> None:
        self.chunks = iter(chunks)
        self.close_error = close_error
        self.closed = False

    def read(self, _size):
        return next(self.chunks, b"")

    def close(self):
        self.closed = True
        if self.close_error:
            raise OSError("synthetic close failure")


def test_bounded_capture_counts_full_stream_and_contains_close_failure() -> None:
    stream = _ChunkStream([b"abcd", b"efgh"], close_error=True)
    capture = scheduler_module._BoundedCapture(stream, limit=5)
    capture.join()
    assert bytes(capture.data) == b"abcde"
    assert capture.total_bytes == 8 and capture.truncated is True and stream.closed


class _Process:
    def __init__(self, *, poll=None, wait_error=None) -> None:
        self.pid = 4242
        self._handle = 73
        self.poll_value = poll
        self.wait_error = wait_error
        self.wait_calls = []

    def poll(self):
        return self.poll_value

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self.wait_error is not None:
            raise self.wait_error
        self.poll_value = -9
        return self.poll_value


class _Closer:
    def __init__(self, *, error=False) -> None:
        self.error = error
        self.calls = []

    def close(self):
        self.calls.append("close")
        if self.error:
            raise OSError("synthetic close failure")

    def join(self, timeout=None):
        self.calls.append(("join", timeout))


def _owned(*, process=None, windows_job=None):
    value = object.__new__(OwnedWorkerProcess)
    value.process = process or _Process(poll=0)
    value.job_id = INSTANCE_ID
    value.spawn_nonce = "a" * 64
    value._windows_job = windows_job
    value._container_terminated = False
    value._cancel_reason = None
    value._cancel_writer = _Closer()
    value._request_writer = _Closer()
    value._reader = _Closer()
    value.stdout = _Closer()
    value.stderr = _Closer()
    value._protocol_state = SimpleNamespace(ready=True, terminal=False, acknowledge=lambda row: row)
    value._protocol_eof = False
    return value


def test_owned_worker_properties_and_cancellation_are_closed(monkeypatch) -> None:
    owned = _owned()
    written = []
    monkeypatch.setattr(scheduler_module, "write_frame", lambda stream, row: written.append(dict(row)))
    assert owned.ready is True and owned.terminal is False and owned.poll() == 0
    assert owned.protocol_eof is False
    owned.request_cancel("timeout")
    owned.request_cancel("service_shutdown")
    assert written == [{"reason_code": "timeout"}]
    assert owned._cancel_writer.calls == ["close"]
    fresh = _owned()
    with pytest.raises(ValueError, match="unsupported"):
        fresh.request_cancel("unknown")


def test_owned_worker_rejects_nonce_and_missing_windows_container(monkeypatch) -> None:
    owned = _owned()
    with pytest.raises(WorkerProtocolError, match="nonce"):
        owned.terminate_owned("b" * 64)
    monkeypatch.setattr(scheduler_module.os, "name", "nt")
    with pytest.raises(WorkerProtocolError, match="no retained Job Object"):
        owned.terminate_owned(owned.spawn_nonce)


def test_owned_worker_contains_group_loss_and_wait_timeout(monkeypatch) -> None:
    process = _Process(poll=None)
    owned = _owned(process=process)
    monkeypatch.setattr(
        scheduler_module.os,
        "killpg",
        lambda *a: (_ for _ in ()).throw(ProcessLookupError("missing group")),
    )
    with pytest.raises(ProcessLookupError):
        owned.terminate_owned(owned.spawn_nonce)

    process = _Process(
        poll=None,
        wait_error=subprocess.TimeoutExpired(cmd="worker", timeout=5),
    )
    owned = _owned(process=process)
    monkeypatch.setattr(scheduler_module.os, "killpg", lambda *a: None)
    with pytest.raises(OSError, match="leader did not exit"):
        owned.terminate_owned(owned.spawn_nonce)


def test_owned_worker_close_finishes_all_cleanup_before_reraising(monkeypatch) -> None:
    owned = _owned(windows_job=41)
    owned._request_writer = _Closer(error=True)
    owned._cancel_writer = _Closer(error=True)
    owned.terminate_owned = lambda nonce: (_ for _ in ()).throw(WorkerProtocolError("lost"))
    closed = []
    monkeypatch.setattr(
        scheduler_module.ctypes,
        "windll",
        SimpleNamespace(kernel32=SimpleNamespace(CloseHandle=lambda handle: closed.append(handle))),
        raising=False,
    )
    with pytest.raises(WorkerProtocolError, match="lost"):
        owned.close()
    assert closed == [41] and owned._windows_job is None
    assert owned._reader.calls == [("join", 1)]
    assert owned.stdout.calls == [("join", None)]
    assert owned.stderr.calls == [("join", None)]


def test_owned_worker_protocol_reader_retains_failure_and_close_error(monkeypatch) -> None:
    owned = _owned()
    owned._messages = scheduler_module.queue.Queue()
    owned._protocol_reader = _Closer(error=True)
    monkeypatch.setattr(
        scheduler_module,
        "read_frame",
        lambda *_a, **_k: (_ for _ in ()).throw(
            WorkerProtocolError("synthetic protocol failure")
        ),
    )

    owned._read_protocol()

    retained = owned.messages()
    assert len(retained) == 1
    assert isinstance(retained[0], WorkerProtocolError)


def test_owned_worker_windows_termination_failure_is_not_claimed(monkeypatch) -> None:
    owned = _owned(windows_job=41)
    monkeypatch.setattr(scheduler_module.os, "name", "nt")
    monkeypatch.setattr(
        scheduler_module.ctypes,
        "windll",
        SimpleNamespace(
            kernel32=SimpleNamespace(TerminateJobObject=lambda *_a: 0)
        ),
        raising=False,
    )

    with pytest.raises(OSError, match="TerminateJobObject"):
        owned.terminate_owned(owned.spawn_nonce)

    assert owned._container_terminated is False


class _Function:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        self.calls.append(args)
        return self.result


class _Structure:
    _fields_ = []

    def __init__(self):
        for name, kind in self._fields_:
            setattr(self, name, kind() if isinstance(kind, type) and issubclass(kind, _Structure) else 0)


def _ctypes(*, create=41, configure=1, assign=1):
    kernel = SimpleNamespace(
        CreateJobObjectW=_Function(create),
        SetInformationJobObject=_Function(configure),
        AssignProcessToJobObject=_Function(assign),
        TerminateJobObject=_Function(1),
        CloseHandle=_Function(1),
    )
    scalar = object()
    module = SimpleNamespace(
        windll=SimpleNamespace(kernel32=kernel),
        Structure=_Structure,
        c_void_p=lambda value=None: value,
        c_int=scalar,
        c_uint32=scalar,
        c_ulonglong=scalar,
        c_longlong=scalar,
        c_size_t=scalar,
        byref=lambda value: value,
        sizeof=lambda value: 256,
    )
    return module, kernel


def test_windows_job_assignment_configures_kill_on_close(monkeypatch) -> None:
    ctypes, kernel = _ctypes()
    monkeypatch.setattr(scheduler_module, "ctypes", ctypes)
    assert scheduler_module._assign_windows_job(_Process()) == 41
    info = kernel.SetInformationJobObject.calls[0][2]
    assert info.BasicLimitInformation.LimitFlags == 0x00002000
    assert kernel.AssignProcessToJobObject.calls == [(41, 73)]


@pytest.mark.parametrize(
    ("create", "configure", "assign", "message", "closed"),
    [
        (0, 1, 1, "CreateJobObjectW", False),
        (41, 0, 1, "SetInformationJobObject", True),
        (41, 1, 0, "AssignProcessToJobObject", True),
    ],
)
def test_windows_job_assignment_closes_failed_container(monkeypatch, create, configure, assign, message, closed) -> None:
    ctypes, kernel = _ctypes(create=create, configure=configure, assign=assign)
    monkeypatch.setattr(scheduler_module, "ctypes", ctypes)
    with pytest.raises(OSError, match=message):
        scheduler_module._assign_windows_job(_Process())
    assert bool(kernel.CloseHandle.calls) is closed


def _snapshot_with_input(staging: Path):
    payload = b"fixed"
    descriptor = {
        "role": "dataset_train",
        "artifact_id": INSTANCE_ID,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "path": "inputs/data.bin",
    }
    snapshot = copy.deepcopy(SNAPSHOT)
    snapshot["inputs"] = [descriptor]
    return snapshot, payload


def test_staging_verifier_rejects_missing_mismatched_and_linked_inputs(tmp_path: Path) -> None:
    staging = tmp_path / "missing"
    staging.mkdir()
    snapshot, payload = _snapshot_with_input(staging)
    with pytest.raises(OSError, match="inputs are missing"):
        WorkerController._verify_staging_inputs(staging, snapshot)

    staging = tmp_path / "mismatched"
    inputs = staging / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "other.bin").write_bytes(payload)
    with pytest.raises(OSError, match="differ from the snapshot"):
        WorkerController._verify_staging_inputs(staging, snapshot)

    staging = tmp_path / "linked"
    inputs = staging / "inputs"
    inputs.mkdir(parents=True)
    target = tmp_path / "linked-target.bin"
    target.write_bytes(payload)
    (inputs / "data.bin").symlink_to(target)
    with pytest.raises(OSError, match="not a confined regular file"):
        WorkerController._verify_staging_inputs(staging, snapshot)


def test_staging_verifier_rejects_linked_roots_and_post_open_type_change(
    tmp_path: Path, monkeypatch
) -> None:
    real_staging = tmp_path / "real-staging"
    real_staging.mkdir()
    linked_staging = tmp_path / "linked-staging"
    linked_staging.symlink_to(real_staging, target_is_directory=True)
    snapshot = copy.deepcopy(SNAPSHOT)
    snapshot["inputs"] = []
    with pytest.raises(OSError, match="staging directory is not confined"):
        WorkerController._verify_staging_inputs(linked_staging, snapshot)

    staging = tmp_path / "linked-inputs-root"
    staging.mkdir()
    outside = tmp_path / "outside-inputs"
    outside.mkdir()
    (staging / "inputs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError, match="inputs directory is not confined"):
        WorkerController._verify_staging_inputs(staging, snapshot)

    staging = tmp_path / "post-open-change"
    inputs = staging / "inputs"
    inputs.mkdir(parents=True)
    snapshot, payload = _snapshot_with_input(staging)
    (inputs / "data.bin").write_bytes(payload)
    real_fstat = os.fstat
    monkeypatch.setattr(
        scheduler_module.os,
        "fstat",
        lambda fd: SimpleNamespace(st_mode=scheduler_module.stat.S_IFDIR),
    )
    with pytest.raises(OSError, match="changed before verification"):
        WorkerController._verify_staging_inputs(staging, snapshot)
    monkeypatch.setattr(scheduler_module.os, "fstat", real_fstat)


def test_worker_launch_failure_closes_every_created_pipe(monkeypatch, tmp_path: Path) -> None:
    controller = WorkerController()
    real_close = os.close
    closed = []

    def recorded_close(fd):
        real_close(fd)
        closed.append(fd)
        if len(closed) == 1:
            raise OSError("synthetic post-close failure")

    monkeypatch.setattr(scheduler_module.os, "close", recorded_close)
    monkeypatch.setattr(
        scheduler_module.subprocess,
        "Popen",
        lambda *a, **k: (_ for _ in ()).throw(OSError("synthetic launch failure")),
    )
    with pytest.raises(OSError, match="synthetic launch failure"):
        controller.launch(
            job_id=INSTANCE_ID,
            instance_id=INSTANCE_ID,
            runtime_profile="wsl-cpu",
            root=tmp_path,
            request={"operation": "model_prepare"},
            request_sha256="a" * 64,
            schema_id="ModelPrepareRequest",
            input_snapshot=SNAPSHOT,
            input_snapshot_sha256="b" * 64,
            output_allocations=ALLOCATIONS,
        )
    assert len(set(closed)) == 6


def test_worker_launch_failure_after_spawn_kills_process_and_revokes_lease(
    monkeypatch, tmp_path: Path
) -> None:
    class Spawned:
        def __init__(self) -> None:
            self.pid = 4242
            self.stdout = None
            self.stderr = None
            self.killed = False
            self.waited = False

        def poll(self):
            return None

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            assert timeout == 5
            self.waited = True
            return -9

    lease_fd = os.open(os.devnull, os.O_RDONLY)
    lease = SimpleNamespace(make_inheritable=lambda: lease_fd)
    controller = WorkerController(instance_lease=lease)
    process = Spawned()
    monkeypatch.setattr(scheduler_module.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(
        scheduler_module,
        "OwnedWorkerProcess",
        lambda **_values: (_ for _ in ()).throw(
            RuntimeError("synthetic ownership construction failure")
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="ownership construction"):
            controller.launch(
                job_id=INSTANCE_ID,
                instance_id=INSTANCE_ID,
                runtime_profile="wsl-cpu",
                root=tmp_path,
                request={"operation": "model_prepare"},
                request_sha256="a" * 64,
                schema_id="ModelPrepareRequest",
                input_snapshot=SNAPSHOT,
                input_snapshot_sha256="b" * 64,
                output_allocations=ALLOCATIONS,
            )
        assert process.killed is True and process.waited is True
        assert os.get_inheritable(lease_fd) is False
    finally:
        os.close(lease_fd)


@pytest.fixture
def lifecycle_runtime(tmp_path: Path):
    database = Database(tmp_path / "lifecycle")
    database.initialize()
    registry = Registry(database, database.root, INSTANCE_ID, b"c" * 32)
    scheduler = Scheduler(
        database,
        registry,
        database.root,
        INSTANCE_ID,
        INSTALLATION_ID,
        "wsl-cpu",
        "0.1.0",
        {
            "model_prepare": {"available": True},
            "tiny_train": {"available": True},
        },
        admission_planner=TinyReservation(),
        clock=Clock(),
    )
    try:
        yield database, registry, scheduler
    finally:
        scheduler._clear_active()
        database.close()


def test_recovery_cursor_mismatch_enters_read_only_recovery(lifecycle_runtime) -> None:
    database, _, scheduler = lifecycle_runtime
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    with database.transaction() as connection:
        row = connection.execute(
            "SELECT record_json FROM jobs WHERE job_id = ?",
            (accepted["job_id"],),
        ).fetchone()
        record = json.loads(row[0])
        record["last_cursor"] += 1
        connection.execute(
            "UPDATE jobs SET record_json = ? WHERE job_id = ?",
            (canonical_json(record).decode(), accepted["job_id"]),
        )

    assert scheduler.recover() == 0
    assert database.read_only is True
    assert scheduler._scheduling_paused is True


def test_dispatch_rejects_changed_request_artifact_digest(
    lifecycle_runtime, monkeypatch
) -> None:
    _, registry, scheduler = lifecycle_runtime
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    monkeypatch.setattr(registry, "read_artifact_bytes", lambda _artifact_id: b"{}")

    failed = scheduler.dispatch_once()

    assert failed["job_id"] == accepted["job_id"]
    assert failed["state"] == "failed"
    assert failed["error"]["code"] == "WORKER_PROTOCOL_ERROR"


def test_dispatch_rolls_back_when_training_target_is_corrupt(lifecycle_runtime) -> None:
    _, _, scheduler = lifecycle_runtime
    accepted = scheduler.submit(tiny_train(), str(uuid.uuid4())).job
    scheduler.training_store = SimpleNamespace(
        get_job_context=lambda _job_id: SimpleNamespace(
            resolved={"requested_final_step": 0}
        )
    )

    with pytest.raises(ApiError) as raised:
        scheduler.dispatch_once()

    assert raised.value.reason_code == "STORAGE_CORRUPT"
    assert scheduler.get(accepted["job_id"])["state"] == "queued"


@pytest.mark.parametrize(
    "tail",
    [
        {"type": "error"},
        {"type": "event"},
    ],
)
def test_tick_rejects_second_terminal_or_data_after_terminal(
    lifecycle_runtime, tail
) -> None:
    _, _, scheduler = lifecycle_runtime
    worker = ScriptedWorker(
        [
            {"type": "ready"},
            {"type": "result", "result": {"accepted": True}},
            tail,
        ],
        None,
        protocol_eof=False,
    )
    scheduler.worker_controller = ScriptedController(worker)
    scheduler.submit(model_prepare(), str(uuid.uuid4()))
    scheduler.dispatch_once()

    failed = scheduler.tick()

    assert failed["state"] == "failed"
    assert failed["error"]["code"] == "WORKER_PROTOCOL_ERROR"
    assert worker.terminated == 1
    assert worker.closed == 1


def test_timeout_cancel_terminates_after_exact_grace_boundary(lifecycle_runtime) -> None:
    _, _, scheduler = lifecycle_runtime
    worker = ScriptedWorker(
        [{"type": "ready"}],
        None,
        protocol_eof=False,
    )
    scheduler.worker_controller = ScriptedController(worker)
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    scheduler.dispatch_once()
    running = scheduler.tick()
    assert running["state"] == "running"

    later = [
        scheduler_module._parse_time(running["started_at"])
        + scheduler_module.timedelta(seconds=scheduler_module.DEADLINE_SECONDS + 1)
    ]
    scheduler._clock = lambda: later[0]
    cancelling = scheduler.tick()
    assert cancelling["state"] == "cancelling"
    assert worker.cancelled == ["timeout"]

    later[0] += scheduler_module.timedelta(
        seconds=scheduler_module.CANCEL_GRACE_SECONDS + 1
    )
    interrupted = scheduler.tick()
    assert interrupted["job_id"] == accepted["job_id"]
    assert interrupted["state"] == "interrupted"
    assert interrupted["terminal_reason"] == "timeout"
    assert worker.terminated == 1
    assert worker.closed == 1


def test_clean_worker_exit_without_terminal_message_fails_job(lifecycle_runtime) -> None:
    _, _, scheduler = lifecycle_runtime
    worker = ScriptedWorker([{"type": "ready"}, None], 0)
    scheduler.worker_controller = ScriptedController(worker)
    scheduler.submit(model_prepare(), str(uuid.uuid4()))
    scheduler.dispatch_once()

    failed = scheduler.tick()

    assert failed["state"] == "failed"
    assert failed["error"]["code"] == "WORKER_PROTOCOL_ERROR"
    assert worker.terminated == 1


@pytest.mark.parametrize("with_operation_store", [False, True])
def test_result_after_cancellation_is_rejected_for_both_terminal_paths(
    lifecycle_runtime, with_operation_store
) -> None:
    _, _, scheduler = lifecycle_runtime
    worker = ScriptedWorker([{"type": "ready"}], None, protocol_eof=False)
    scheduler.worker_controller = ScriptedController(worker)
    if with_operation_store:
        scheduler.operation_store = ScriptedOperationStore()
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    scheduler.dispatch_once()
    assert scheduler.tick()["state"] == "running"
    assert scheduler.cancel(accepted["job_id"])["state"] == "cancelling"
    worker._messages.append(
        [{"type": "result", "result": {"accepted": True}}, None]
    )
    worker._exit_code = 0
    worker.protocol_eof = True

    failed = scheduler.tick()

    assert failed["state"] == "failed"
    assert failed["error"]["code"] == "WORKER_PROTOCOL_ERROR"


def test_interruption_must_match_committed_checkpoint_boundary(
    lifecycle_runtime,
) -> None:
    _, _, scheduler = lifecycle_runtime
    worker = ScriptedWorker([{"type": "ready"}], None, protocol_eof=False)
    scheduler.worker_controller = ScriptedController(worker)
    scheduler.operation_store = ScriptedOperationStore()
    scheduler.submit(model_prepare(), str(uuid.uuid4()))
    scheduler.dispatch_once()
    assert scheduler.tick()["state"] == "running"
    worker._messages.append(
        [
            {
                "type": "interrupted",
                "reason_code": "user_cancelled",
                "checkpoint_id": INSTANCE_ID,
                "checkpoint_step": 0,
                "error": None,
            },
            None,
        ]
    )
    worker._exit_code = 0
    worker.protocol_eof = True

    failed = scheduler.tick()

    assert failed["state"] == "failed"
    assert failed["error"]["code"] == "WORKER_PROTOCOL_ERROR"


def test_artifact_rejection_and_acknowledgment_failure_are_bounded(
    lifecycle_runtime,
) -> None:
    _, _, scheduler = lifecycle_runtime
    worker = ScriptedWorker([{"type": "ready"}], None, protocol_eof=False)
    store = ScriptedOperationStore()
    scheduler.worker_controller = ScriptedController(worker)
    scheduler.operation_store = store
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    scheduler.dispatch_once()
    assert scheduler.tick()["state"] == "running"
    proposal = {
        "type": "artifact_ready",
        "role": "preview",
        "sha256": "d" * 64,
    }

    def fail_prepare(_job_id, _proposal):
        raise RuntimeError("synthetic preparation failure")

    store.prepare_artifact = fail_prepare
    scheduler._handle_worker_message(proposal)
    assert worker.acks[-1]["type"] == "commit_rejected"
    assert worker.acks[-1]["error"]["code"] == "INTERNAL_ERROR"

    store.prepare_artifact = lambda _job_id, value: dict(value)
    worker.send_ack = lambda _message: (_ for _ in ()).throw(
        OSError("synthetic acknowledgment failure")
    )
    with pytest.raises(WorkerProtocolError, match="acknowledgment"):
        scheduler._handle_worker_message(proposal)
    assert scheduler.get(accepted["job_id"])["state"] == "running"


def test_cancelling_checkpoint_rejection_uses_specific_job_error(
    lifecycle_runtime,
) -> None:
    _, _, scheduler = lifecycle_runtime
    worker = ScriptedWorker([{"type": "ready"}], None, protocol_eof=False)
    scheduler.worker_controller = ScriptedController(worker)
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    scheduler.dispatch_once()
    assert scheduler.tick()["state"] == "running"
    assert scheduler.cancel(accepted["job_id"])["state"] == "cancelling"

    error = scheduler._proposal_error(
        "checkpoint", RuntimeError("synthetic checkpoint failure")
    )

    assert error["code"] == "CHECKPOINT_WRITE_FAILED"


@pytest.mark.parametrize("proposal_type", ["artifact_ready", "checkpoint_ready"])
def test_worker_proposal_without_operation_store_fails_job(
    lifecycle_runtime, proposal_type
) -> None:
    _, _, scheduler = lifecycle_runtime
    worker = ScriptedWorker(
        [{"type": "ready"}, {"type": proposal_type}],
        None,
        protocol_eof=False,
    )
    scheduler.worker_controller = ScriptedController(worker)
    scheduler.submit(model_prepare(), str(uuid.uuid4()))
    scheduler.dispatch_once()

    failed = scheduler.tick()

    assert failed["state"] == "failed"
    assert failed["error"]["code"] == "WORKER_PROTOCOL_ERROR"
    assert worker.terminated == 1


def test_committed_checkpoint_survives_acknowledgment_loss(
    lifecycle_runtime,
) -> None:
    _, _, scheduler = lifecycle_runtime
    worker = ScriptedWorker([{"type": "ready"}], None, protocol_eof=False)
    store = ScriptedOperationStore()
    scheduler.worker_controller = ScriptedController(worker)
    scheduler.operation_store = store
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    scheduler.dispatch_once()
    assert scheduler.tick()["state"] == "running"
    worker.send_ack = lambda _message: (_ for _ in ()).throw(
        OSError("synthetic acknowledgment loss")
    )

    with pytest.raises(WorkerProtocolError, match="checkpoint acknowledgment"):
        scheduler._commit_worker_checkpoint(
            accepted["job_id"],
            {"type": "checkpoint_ready", "manifest_sha256": "d" * 64},
        )

    retained = scheduler.get(accepted["job_id"])
    assert retained["checkpoint_boundary"] == {
        "checkpoint_id": "123e4567-e89b-42d3-a456-426614174003",
        "step": 0,
    }
    assert any(call[0] == "complete_checkpoint" for call in store.calls)


def test_exit_with_lost_ownership_pauses_future_scheduling(
    lifecycle_runtime,
) -> None:
    _, _, scheduler = lifecycle_runtime
    worker = ScriptedWorker(
        [{"type": "ready"}, None],
        0,
        terminate_error=OSError("synthetic ownership loss"),
    )
    scheduler.worker_controller = ScriptedController(worker)
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    scheduler.dispatch_once()

    interrupted = scheduler.tick()

    assert interrupted["job_id"] == accepted["job_id"]
    assert interrupted["state"] == "interrupted"
    assert interrupted["terminal_reason"] == "worker_ownership_unknown"
    assert scheduler._scheduling_paused is True


def test_cancelled_worker_exit_without_terminal_is_worker_lost(
    lifecycle_runtime,
) -> None:
    _, _, scheduler = lifecycle_runtime
    worker = ScriptedWorker([{"type": "ready"}], None, protocol_eof=False)
    scheduler.worker_controller = ScriptedController(worker)
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    scheduler.dispatch_once()
    assert scheduler.tick()["state"] == "running"
    assert scheduler.cancel(accepted["job_id"])["state"] == "cancelling"
    worker._messages.append([None])
    worker._exit_code = 0
    worker.protocol_eof = True

    interrupted = scheduler.tick()

    assert interrupted["state"] == "interrupted"
    assert interrupted["terminal_reason"] == "worker_lost"


def test_dispatch_rejects_nonobject_canonical_request_artifact(
    lifecycle_runtime, monkeypatch
) -> None:
    database, registry, scheduler = lifecycle_runtime
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    replacement = b"[]"
    replacement_sha256 = hashlib.sha256(replacement).hexdigest()
    with database.transaction() as connection:
        connection.execute(
            "UPDATE jobs SET request_sha256 = ? WHERE job_id = ?",
            (replacement_sha256, accepted["job_id"]),
        )
    monkeypatch.setattr(
        registry, "read_artifact_bytes", lambda _artifact_id: replacement
    )

    failed = scheduler.dispatch_once()

    assert failed["state"] == "failed"
    assert failed["error"]["code"] == "WORKER_PROTOCOL_ERROR"


@pytest.mark.parametrize(
    "parent",
    [
        None,
        {"checkpoint_id": INSTANCE_ID, "step": True},
    ],
)
def test_resume_dispatch_rejects_malformed_verified_parent(
    lifecycle_runtime, parent
) -> None:
    _, _, scheduler = lifecycle_runtime
    parent_id = str(uuid.uuid4())
    request = {
        "operation": "tiny_resume",
        "checkpoint_id": parent_id,
        "additional_steps": 1,
    }

    class SnapshotStore:
        def get_job_context(self, _job_id):
            return SimpleNamespace(resolved={"requested_final_step": 2})

        def materialize_worker_snapshot(self, _job_id):
            return SimpleNamespace(
                value={
                    "ids": dict(ALLOCATIONS),
                    "resolved": {"parent_checkpoint": parent},
                },
                sha256="e" * 64,
            )

    scheduler.capabilities["tiny_resume"] = {"available": True}
    scheduler.training_store = SnapshotStore()
    scheduler.submit(request, str(uuid.uuid4()))

    failed = scheduler.dispatch_once()

    assert failed["state"] == "failed"
    assert failed["error"]["code"] == "WORKER_PROTOCOL_ERROR"
    assert failed["error"]["message"] == "The fixed worker could not be started."


def test_cancel_is_idempotent_after_interruption_and_conflicts_after_failure(
    lifecycle_runtime,
) -> None:
    _, _, scheduler = lifecycle_runtime
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    interrupted = scheduler.cancel(accepted["job_id"])
    assert interrupted["state"] == "interrupted"
    assert scheduler.cancel(accepted["job_id"])["state"] == "interrupted"

    failed_job = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    scheduler._finish_failed(
        failed_job["job_id"], "WORKER_PROTOCOL_ERROR", "synthetic failure"
    )
    with pytest.raises(ApiError) as conflict:
        scheduler.cancel(failed_job["job_id"])
    assert conflict.value.code == "STATE_CONFLICT"
    with pytest.raises(ValueError, match="user_cancelled"):
        scheduler.cancel(failed_job["job_id"], reason="timeout")


def test_worker_state_machine_rejects_events_before_ready_and_duplicate_ready(
    lifecycle_runtime,
) -> None:
    _, _, scheduler = lifecycle_runtime
    worker = ScriptedWorker([], None, protocol_eof=False)
    scheduler.worker_controller = ScriptedController(worker)
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    assert scheduler.dispatch_once()["state"] == "starting"

    with pytest.raises(WorkerProtocolError, match="event in an invalid state"):
        scheduler._handle_worker_message(
            {
                "type": "event",
                "event_type": "warning",
                "payload": {"code": "EARLY", "message": "too early"},
            }
        )
    scheduler._handle_worker_message({"type": "ready"})
    assert scheduler.get(accepted["job_id"])["state"] == "running"
    with pytest.raises(WorkerProtocolError, match="ready acknowledgment"):
        scheduler._handle_worker_message({"type": "ready"})


def test_delayed_ready_after_cancellation_cannot_restart_job(
    lifecycle_runtime,
) -> None:
    _, _, scheduler = lifecycle_runtime
    worker = ScriptedWorker([], None, protocol_eof=False)
    scheduler.worker_controller = ScriptedController(worker)
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    scheduler.dispatch_once()
    assert scheduler.cancel(accepted["job_id"])["state"] == "cancelling"

    scheduler._handle_worker_message({"type": "ready"})

    assert scheduler.get(accepted["job_id"])["state"] == "cancelling"


def test_training_progress_cannot_exceed_admitted_final_step(
    lifecycle_runtime,
) -> None:
    database, _, scheduler = lifecycle_runtime
    worker = ScriptedWorker([{"type": "ready"}], None, protocol_eof=False)
    scheduler.worker_controller = ScriptedController(worker)
    accepted = scheduler.submit(tiny_train(), str(uuid.uuid4())).job
    set_requested_final_step(database, accepted["job_id"], 3)
    scheduler.dispatch_once()
    assert scheduler.tick()["state"] == "running"

    with pytest.raises(WorkerProtocolError, match="admitted steps"):
        scheduler._handle_worker_message(
            {
                "type": "event",
                "event_type": "progress",
                "payload": {
                    "current": 4,
                    "total": 3,
                    "unit": "updates",
                    "message": "too far",
                },
            }
        )
    assert scheduler.get(accepted["job_id"])["step"] == 0


def test_warning_retention_is_bounded_without_exceeding_event_cardinality(
    lifecycle_runtime,
) -> None:
    _, _, scheduler = lifecycle_runtime
    worker = ScriptedWorker([{"type": "ready"}], None, protocol_eof=False)
    scheduler.worker_controller = ScriptedController(worker)
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    scheduler.dispatch_once()
    assert scheduler.tick()["state"] == "running"

    for index in range(33):
        scheduler._handle_worker_message(
            {
                "type": "event",
                "event_type": "warning",
                "payload": {
                    "code": "TEST_WARNING",
                    "message": f"warning {index}",
                },
            }
        )

    retained = scheduler.get(accepted["job_id"])
    assert len(retained["warnings"]) == 32
    assert retained["warning_suppressed_count"] == 1
    warning_events = [
        event
        for event in scheduler.list_events(accepted["job_id"], 0, 100).items
        if event["event_type"] == "warning"
    ]
    assert len(warning_events) == 32


def test_proposal_rejection_delivery_failure_becomes_protocol_failure(
    lifecycle_runtime,
) -> None:
    _, _, scheduler = lifecycle_runtime
    worker = ScriptedWorker([{"type": "ready"}], None, protocol_eof=False)
    store = ScriptedOperationStore()
    scheduler.worker_controller = ScriptedController(worker)
    scheduler.operation_store = store
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    scheduler.dispatch_once()
    assert scheduler.tick()["state"] == "running"
    store.prepare_artifact = lambda *_a: (_ for _ in ()).throw(
        RuntimeError("synthetic proposal failure")
    )
    worker.send_ack = lambda _message: (_ for _ in ()).throw(
        OSError("synthetic rejection delivery failure")
    )

    with pytest.raises(WorkerProtocolError, match="rejection could not be delivered"):
        scheduler._prepare_worker_artifact(
            accepted["job_id"],
            {"type": "artifact_ready", "role": "preview", "sha256": "f" * 64},
        )


def test_protocol_reader_failure_with_lost_ownership_pauses_scheduler(
    lifecycle_runtime,
) -> None:
    _, _, scheduler = lifecycle_runtime
    worker = ScriptedWorker(
        [WorkerProtocolError("synthetic reader failure")],
        None,
        protocol_eof=False,
        terminate_error=OSError("synthetic ownership loss"),
    )
    scheduler.worker_controller = ScriptedController(worker)
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    scheduler.dispatch_once()

    interrupted = scheduler.tick()

    assert interrupted["job_id"] == accepted["job_id"]
    assert interrupted["state"] == "interrupted"
    assert interrupted["terminal_reason"] == "worker_ownership_unknown"
    assert scheduler._scheduling_paused is True


def test_shutdown_requests_service_cancel_and_stops_after_terminal_tick(
    lifecycle_runtime, monkeypatch
) -> None:
    _, _, scheduler = lifecycle_runtime
    worker = ScriptedWorker([{"type": "ready"}], None, protocol_eof=False)
    scheduler.worker_controller = ScriptedController(worker)
    scheduler.submit(model_prepare(), str(uuid.uuid4()))
    scheduler.dispatch_once()
    assert scheduler.tick()["state"] == "running"
    ticks = []

    def terminal_tick():
        ticks.append("tick")
        return {"state": "interrupted"}

    monkeypatch.setattr(scheduler, "tick", terminal_tick)

    scheduler.shutdown()

    assert worker.cancelled == ["service_shutdown"]
    assert ticks == ["tick"]


def test_scheduler_clock_accepts_text_and_epoch_and_rejects_unknown_type(
    lifecycle_runtime,
) -> None:
    _, _, scheduler = lifecycle_runtime
    scheduler._clock = lambda: "2026-10-01T12:00:00Z"
    assert scheduler._clock_value() == scheduler_module._parse_time(
        "2026-10-01T12:00:00Z"
    )

    scheduler._clock = lambda: 0
    epoch = scheduler._clock_value()
    assert epoch.timestamp() == 0 and epoch.tzinfo is not None

    scheduler._clock = lambda: object()
    with pytest.raises(TypeError, match="clock must return"):
        scheduler._clock_value()
