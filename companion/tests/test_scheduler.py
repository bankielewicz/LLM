from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from llm_foundations_companion.database import Database
from llm_foundations_companion.errors import ApiError
from llm_foundations_companion.registry import Registry
from llm_foundations_companion.scheduler import IdempotencyStore, Scheduler
from llm_foundations_companion.schema import canonical_json, validate_schema
from llm_foundations_companion.worker_protocol import WorkerProtocolError


INSTANCE_ID = "123e4567-e89b-42d3-a456-426614174000"
INSTALLATION_ID = "123e4567-e89b-42d3-a456-426614174001"
DATASET_ID = "123e4567-e89b-42d3-a456-426614174002"
CHECKPOINT_ID = "123e4567-e89b-42d3-a456-426614174003"
PREVIEW_ARTIFACT_ID = "123e4567-e89b-42d3-a456-426614174004"


class TinyReservation:
    def plan(self, request):
        return {
            "byte_count": 1,
            "artifact_rows": 1,
            "dataset_rows": 0,
            "run_rows": 0,
            "model_rows": 0,
            "checkpoint_rows": 0,
        }


class ScriptedWorker:
    def __init__(
        self,
        messages,
        exit_code,
        *,
        protocol_eof=True,
        cancel_error=None,
        terminate_error=None,
    ):
        self.process = SimpleNamespace(pid=4242)
        self.spawn_nonce = "f" * 64
        self.protocol_eof = protocol_eof
        self._messages = deque((list(messages),))
        self._exit_code = exit_code
        self._cancel_error = cancel_error
        self._terminate_error = terminate_error
        self.terminated = 0
        self.closed = 0
        self.cancelled = []

    def messages(self):
        return self._messages.popleft() if self._messages else []

    def poll(self):
        return self._exit_code

    def request_cancel(self, reason):
        if self._cancel_error is not None:
            raise self._cancel_error
        self.cancelled.append(reason)

    def terminate_owned(self, expected_nonce):
        assert expected_nonce == self.spawn_nonce
        if self._terminate_error is not None:
            raise self._terminate_error
        self.terminated += 1

    def close(self):
        self.closed += 1


class ScriptedController:
    def __init__(self, worker):
        self.worker = worker

    def launch(self, **_values):
        return self.worker


class Clock:
    def __init__(self):
        self.value = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        self.value += timedelta(milliseconds=1)
        return self.value


def model_prepare():
    return {
        "operation": "model_prepare",
        "model_profile_id": "smollm2-135m-instruct-v1",
        "accept_download": True,
    }


def tokenizer(seed=17):
    return {
        "operation": "tokenizer_train",
        "dataset_id": DATASET_ID,
        "tokenizer_profile_id": "byte-v1",
        "vocab_size": 257,
        "seed": seed,
    }


def generate():
    return {
        "operation": "generate",
        "checkpoint_id": CHECKPOINT_ID,
        "prompt": "hello",
        "max_new_tokens": 8,
        "temperature": 1,
        "top_p": 1,
        "seed": 17,
        "preview_artifact_id": PREVIEW_ARTIFACT_ID,
        "context_preview_digest": "a" * 64,
    }


@pytest.fixture
def runtime(tmp_path):
    db = Database(tmp_path / "storage")
    db.initialize()
    registry = Registry(db, db.root, INSTANCE_ID, b"c" * 32)
    scheduler = Scheduler(
        db,
        registry,
        db.root,
        INSTANCE_ID,
        INSTALLATION_ID,
        "wsl-cpu",
        "0.1.0",
        {
            operation: {"available": True}
            for operation in ("model_prepare", "tokenizer_train", "generate")
        },
        admission_planner=TinyReservation(),
        clock=Clock(),
    )
    try:
        yield db, registry, scheduler
    finally:
        scheduler.shutdown()
        db.close()


def counts(db):
    with db.read() as connection:
        return tuple(
            int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("jobs", "job_events", "artifacts", "reservations", "idempotency_records")
        )


def test_submit_is_one_atomic_job_artifact_reservation_event_and_replay(runtime):
    db, registry, scheduler = runtime
    key = str(uuid.uuid4())
    first = scheduler.submit(model_prepare(), key)
    assert first.status_code == 202
    assert first.replayed is False
    job = first.job
    assert job["state"] == "queued"
    assert job["last_cursor"] == 1
    assert job["queue_position"] == 1
    assert validate_schema(
        {"$ref": "#/components/schemas/Job"}, job, document="openapi.json"
    ) == job
    descriptor = registry.get_artifact(job["request"]["artifact_id"])
    assert descriptor["type"] == "job_request"
    assert registry.read_artifact_bytes(descriptor["artifact_id"]) == canonical_json(model_prepare())
    assert counts(db) == (1, 1, 1, 1, 1)

    replay = scheduler.submit(model_prepare(), key)
    assert replay == type(replay)(202, first.response_body, True)
    assert counts(db) == (1, 1, 1, 1, 1)

    with pytest.raises(ApiError) as caught:
        scheduler.submit(tokenizer(seed=18), key)
    assert caught.value.code == "IDEMPOTENCY_CONFLICT"
    assert counts(db) == (1, 1, 1, 1, 1)


def test_submit_canonicalizes_the_schema_normalized_request(runtime):
    _, registry, scheduler = runtime
    request = generate()
    job = scheduler.submit(request, str(uuid.uuid4())).job
    normalized = dict(request, temperature=1.0, top_p=1.0)

    assert request["temperature"] == 1 and isinstance(request["temperature"], int)
    assert request["top_p"] == 1 and isinstance(request["top_p"], int)
    assert registry.read_artifact_bytes(job["request"]["artifact_id"]) == canonical_json(
        normalized
    )
    assert job["request"]["canonical_sha256"] == hashlib.sha256(
        canonical_json(normalized)
    ).hexdigest()


def test_idempotency_is_route_scoped_24_hours_and_never_stores_errors(runtime):
    db, _, scheduler = runtime
    request = model_prepare()
    digest = hashlib.sha256(canonical_json(request)).hexdigest()
    key = str(uuid.uuid4())
    commit = scheduler.idempotency.prepare("POST", "/api/v1/example/a", key, digest)
    with db.transaction() as connection:
        commit.record_success(connection, status_code=201, response_body={"ok": True})
    assert scheduler.idempotency.lookup("POST", "/api/v1/example/a", key, digest).response_body == b'{"ok":true}'
    assert scheduler.idempotency.lookup("POST", "/api/v1/example/b", key, digest) is None
    with pytest.raises(ValueError, match="2xx"):
        with db.transaction() as connection:
            scheduler.idempotency.record_success(
                connection,
                method="POST",
                resolved_path="/api/v1/example/a",
                key=str(uuid.uuid4()),
                request_sha256=digest,
                status_code=400,
                response_body=b"{}",
            )


def test_capability_rejection_changes_no_state_but_exact_replay_precedes_it(tmp_path):
    db = Database(tmp_path / "storage")
    db.initialize()
    registry = Registry(db, db.root, INSTANCE_ID, b"d" * 32)
    scheduler = Scheduler(
        db,
        registry,
        db.root,
        INSTANCE_ID,
        INSTALLATION_ID,
        "wsl-cpu",
        "0.1.0",
        {"model_prepare": {"available": False, "message": "S1 unavailable"}},
        clock=Clock(),
    )
    request, key = model_prepare(), str(uuid.uuid4())
    digest = hashlib.sha256(canonical_json(request)).hexdigest()
    with pytest.raises(ApiError) as caught:
        scheduler.submit(request, key)
    assert caught.value.code == "CAPABILITY_UNAVAILABLE"
    assert counts(db) == (0, 0, 0, 0, 0)

    body = canonical_json({"prior": "accepted"})
    with db.transaction() as connection:
        scheduler.idempotency.record_success(
            connection,
            method="POST",
            resolved_path="/api/v1/jobs",
            key=key,
            request_sha256=digest,
            status_code=202,
            response_body=body,
        )
    replay = scheduler.submit(request, key)
    assert replay.response_body == body and replay.replayed is True


def test_queue_cap_and_queued_cancel_are_atomic_and_release_capacity(runtime):
    db, _, scheduler = runtime
    jobs = [scheduler.submit(model_prepare(), str(uuid.uuid4())).job for _ in range(8)]
    before = counts(db)
    with pytest.raises(ApiError) as caught:
        scheduler.submit(model_prepare(), str(uuid.uuid4()))
    assert caught.value.code == "QUEUE_FULL"
    assert counts(db) == before

    cancelled = scheduler.cancel(jobs[0]["job_id"])
    assert cancelled["state"] == "interrupted"
    assert cancelled["started_at"] is None
    assert cancelled["terminal_reason"] == "user_cancelled"
    assert cancelled["queue_position"] is None
    assert validate_schema(
        {"$ref": "#/components/schemas/Job"}, cancelled, document="openapi.json"
    ) == cancelled
    page = scheduler.events.list(jobs[0]["job_id"], 0, 100)
    assert [event["event_type"] for event in page.items] == [
        "state_changed",
        "state_changed",
        "state_changed",
        "terminal",
    ]
    assert [event["payload"]["state"] for event in page.items[:3]] == [
        "queued",
        "cancelling",
        "interrupted",
    ]
    with db.read() as connection:
        state = connection.execute(
            "SELECT state FROM reservations WHERE reservation_id = ?",
            (jobs[0]["job_id"],),
        ).fetchone()[0]
    assert state == "released"
    assert scheduler.get(jobs[1]["job_id"])["queue_position"] == 1


def test_list_cursor_binds_filters_instance_and_order(runtime):
    _, _, scheduler = runtime
    for _ in range(3):
        scheduler.submit(model_prepare(), str(uuid.uuid4()))
    first = scheduler.list(operation="model_prepare", limit=2)
    assert len(first["items"]) == 2
    assert first["next_cursor"]
    second = scheduler.list(
        operation="model_prepare", cursor=first["next_cursor"], limit=2
    )
    assert len(second["items"]) == 1
    assert {item["job_id"] for item in first["items"]}.isdisjoint(
        {item["job_id"] for item in second["items"]}
    )
    with pytest.raises(ApiError) as caught:
        scheduler.list(state="queued", cursor=first["next_cursor"], limit=2)
    assert caught.value.reason_code == "SEMANTIC_INVALID"


def test_event_replay_bounds_and_immutable_trigger(runtime):
    db, _, scheduler = runtime
    job = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    assert scheduler.events.list(job["job_id"], 1, 100).as_dict() == {
        "items": [],
        "next_after_cursor": 1,
        "has_more": False,
    }
    with pytest.raises(ApiError) as caught:
        scheduler.events.list(job["job_id"], 2, 100)
    assert caught.value.reason_code == "SEMANTIC_INVALID"
    with pytest.raises(sqlite3.IntegrityError):
        with db.transaction() as connection:
            connection.execute(
                "UPDATE job_events SET occurred_at = occurred_at WHERE job_id = ?",
                (job["job_id"],),
            )


def test_recovery_interrupts_prior_active_jobs_and_leaves_queued_fifo(runtime):
    db, _, scheduler = runtime
    active = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    queued = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    with db.transaction() as connection:
        row = connection.execute(
            "SELECT record_json FROM jobs WHERE job_id = ?", (active["job_id"],)
        ).fetchone()
        record = json.loads(row[0])
        now = "2026-10-01T12:00:01.000Z"
        record.update(state="running", started_at=now, updated_at=now, queue_position=None)
        connection.execute(
            "UPDATE jobs SET state='running', updated_at=?, record_json=? WHERE job_id=?",
            (now, canonical_json(record).decode(), active["job_id"]),
        )
        event = scheduler.events.append(
            connection,
            job_id=active["job_id"],
            event_type="state_changed",
            payload={"state": "running", "step": None, "requested_final_step": None},
            occurred_at=now,
        )
        record["last_cursor"] = event["cursor"]
        connection.execute(
            "UPDATE jobs SET record_json=? WHERE job_id=?",
            (canonical_json(record).decode(), active["job_id"]),
        )
    assert scheduler.recover() == 1
    recovered = scheduler.get(active["job_id"])
    assert recovered["state"] == "interrupted"
    assert recovered["terminal_reason"] == "service_restarted"
    assert scheduler.get(queued["job_id"])["state"] == "queued"
    assert scheduler.get(queued["job_id"])["queue_position"] == 1


@pytest.mark.parametrize(
    ("tail", "exit_code", "expected_state"),
    [
        ([], 0, "completed"),
        ([], 7, "failed"),
        ([WorkerProtocolError("data after terminal")], 0, "failed"),
    ],
)
def test_terminal_result_waits_for_clean_protocol_eof_and_zero_exit(
    runtime, tail, exit_code, expected_state
):
    _, _, scheduler = runtime
    worker = ScriptedWorker(
        [
            {"type": "ready"},
            {
                "type": "result",
                "operation": "model_prepare",
                "result": {"accepted": True},
            },
            *tail,
            None,
        ],
        exit_code,
    )
    scheduler.worker_controller = ScriptedController(worker)
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job

    assert scheduler.dispatch_once()["state"] == "starting"
    finished = scheduler.tick()

    assert finished["state"] == expected_state
    assert worker.terminated == 1
    assert worker.closed == 1
    assert scheduler.get(accepted["job_id"])["state"] == expected_state
    terminal = scheduler.events.list(accepted["job_id"], 0, 100).items[-1]
    assert terminal["event_type"] == "terminal"
    assert terminal["payload"]["state"] == expected_state


def test_cancel_pipe_failure_becomes_worker_lost_instead_of_escaping(runtime):
    _, _, scheduler = runtime
    worker = ScriptedWorker(
        [{"type": "ready"}],
        None,
        protocol_eof=False,
        cancel_error=BrokenPipeError(),
    )
    scheduler.worker_controller = ScriptedController(worker)
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    scheduler.dispatch_once()
    assert scheduler.tick()["state"] == "running"

    cancelled = scheduler.cancel(accepted["job_id"])

    assert cancelled["state"] == "interrupted"
    assert cancelled["terminal_reason"] == "worker_lost"
    assert worker.terminated == 1
    assert worker.closed == 1


def test_expected_worker_error_may_exit_nonzero_after_clean_protocol(runtime):
    _, _, scheduler = runtime
    worker = ScriptedWorker(
        [
            {"type": "ready"},
            {
                "type": "error",
                "error": {
                    "code": "WORKER_PROTOCOL_ERROR",
                    "message": "No installed implementation.",
                    "retryable": False,
                    "field_errors": [],
                },
            },
            None,
        ],
        2,
    )
    scheduler.worker_controller = ScriptedController(worker)
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    scheduler.dispatch_once()

    failed = scheduler.tick()

    assert failed["state"] == "failed"
    assert failed["error"]["code"] == "WORKER_PROTOCOL_ERROR"
    assert scheduler.get(accepted["job_id"])["state"] == "failed"


def test_expected_interrupted_terminal_reconciles_after_cancel(runtime):
    _, _, scheduler = runtime
    worker = ScriptedWorker([{"type": "ready"}], None, protocol_eof=False)
    scheduler.worker_controller = ScriptedController(worker)
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    scheduler.dispatch_once()
    assert scheduler.tick()["state"] == "running"
    assert scheduler.cancel(accepted["job_id"])["state"] == "cancelling"
    assert worker.cancelled == ["user_cancelled"]
    worker._messages.append(
        [
            {
                "type": "interrupted",
                "reason_code": "user_cancelled",
                "checkpoint_id": None,
                "checkpoint_step": None,
                "error": None,
            },
            None,
        ]
    )
    worker._exit_code = 0
    worker.protocol_eof = True

    interrupted = scheduler.tick()

    assert interrupted["state"] == "interrupted"
    assert interrupted["terminal_reason"] == "user_cancelled"
    assert worker.terminated == 1
    assert worker.closed == 1


def test_read_only_recovery_shutdown_terminates_without_database_write(runtime):
    db, _, scheduler = runtime
    worker = ScriptedWorker([{"type": "ready"}], None, protocol_eof=False)
    scheduler.worker_controller = ScriptedController(worker)
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    scheduler.dispatch_once()
    assert scheduler.tick()["state"] == "running"
    db.enter_read_only_recovery("STORAGE_CORRUPT")

    scheduler.shutdown()

    assert worker.terminated == 1
    assert worker.closed == 1
    assert scheduler._active is None
    assert scheduler.get(accepted["job_id"])["state"] == "running"


def test_read_only_recovery_shutdown_retains_worker_when_termination_fails(runtime):
    db, _, scheduler = runtime
    worker = ScriptedWorker(
        [{"type": "ready"}],
        None,
        protocol_eof=False,
        terminate_error=OSError("container termination failed"),
    )
    scheduler.worker_controller = ScriptedController(worker)
    scheduler.submit(model_prepare(), str(uuid.uuid4()))
    scheduler.dispatch_once()
    assert scheduler.tick()["state"] == "running"
    db.enter_read_only_recovery("STORAGE_CORRUPT")

    with pytest.raises(OSError, match="container termination failed"):
        scheduler.shutdown()

    assert scheduler._active is worker
    assert worker.closed == 0
    worker._terminate_error = None
    scheduler.shutdown()
