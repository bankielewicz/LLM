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
from llm_foundations_companion.operation_store import OperationStoreError
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
        self.acks = []

    def messages(self):
        return self._messages.popleft() if self._messages else []

    def poll(self):
        return self._exit_code

    def request_cancel(self, reason):
        if self._cancel_error is not None:
            raise self._cancel_error
        self.cancelled.append(reason)

    def send_ack(self, message):
        self.acks.append(dict(message))

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


class ScriptedOperationStore:
    def __init__(self):
        self.calls = []
        self.pending_artifact_ids = (PREVIEW_ARTIFACT_ID,)

    def prepare_artifact(self, job_id, proposal):
        self.calls.append(("prepare_artifact", job_id, dict(proposal)))
        return dict(proposal)

    def commit_artifact_in(self, connection, prepared):
        self.calls.append(("commit_artifact", prepared["role"]))
        return SimpleNamespace(
            ack={
                "type": "artifact_prepared",
                "role": prepared["role"],
                "artifact_id": PREVIEW_ARTIFACT_ID,
                "sha256": prepared["sha256"],
            }
        )

    def prepare_checkpoint(self, job_id, proposal):
        self.calls.append(("prepare_checkpoint", job_id, dict(proposal)))
        return SimpleNamespace(
            step=0,
            manifest_sha256=proposal["manifest_sha256"],
        )

    def commit_checkpoint_in(self, connection, prepared):
        self.calls.append(("commit_checkpoint", prepared.manifest_sha256))
        return SimpleNamespace(
            ack={
                "type": "checkpoint_committed",
                "checkpoint_id": CHECKPOINT_ID,
                "step": 0,
                "sha256": prepared.manifest_sha256,
                "run_id": DATASET_ID,
                "artifact_ids": {
                    name: str(uuid.uuid5(uuid.NAMESPACE_URL, name))
                    for name in (
                        "model.safetensors",
                        "optimizer.safetensors",
                        "rng.safetensors",
                        "tokenizer.json",
                        "config.json",
                        "trainer_state.json",
                        "manifest.json",
                    )
                },
            },
            checkpoint_id=CHECKPOINT_ID,
            step=0,
            sha256=prepared.manifest_sha256,
            artifact_ids=(),
            run_id=DATASET_ID,
        )

    def complete_checkpoint(self, prepared):
        self.calls.append(
            (
                "complete_checkpoint",
                prepared.manifest_sha256,
            )
        )

    def prepare_terminal(
        self, job_id, *, state, reason_code, finished_at, result=None
    ):
        self.calls.append(
            ("prepare_terminal", job_id, state, reason_code, finished_at)
        )
        return SimpleNamespace(
            state=state,
            result=result,
            finished_at=finished_at,
            pending_artifact_ids=self.pending_artifact_ids,
        )

    def finalize_terminal_in(self, connection, prepared):
        self.calls.append(("finalize_terminal", prepared.state))
        return SimpleNamespace(
            result=prepared.result,
            run_id=DATASET_ID if prepared.state == "completed" else None,
            model_id=None,
            artifact_ids=(),
        )

    def complete_terminal(self, prepared):
        self.calls.append(
            (
                "complete_terminal",
                prepared.state,
                prepared.pending_artifact_ids,
            )
        )


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


def tiny_train(steps=3):
    return {
        "operation": "tiny_train",
        "dataset_id": DATASET_ID,
        "tokenizer_id": PREVIEW_ARTIFACT_ID,
        "architecture_profile_id": "tiny-v2-standard-v1",
        "steps": steps,
        "eval_every": 1,
        "batch_size": 1,
        "learning_rate": 0.001,
        "context": 8,
        "width": 16,
        "heads": 1,
        "layers": 1,
        "seed": 17,
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
            for operation in (
                "model_prepare",
                "tokenizer_train",
                "tiny_train",
                "generate",
            )
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


def set_requested_final_step(db, job_id, step):
    with db.transaction() as connection:
        row = connection.execute(
            "SELECT record_json FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        record = json.loads(row[0])
        record["requested_final_step"] = step
        connection.execute(
            "UPDATE jobs SET record_json = ? WHERE job_id = ?",
            (canonical_json(record).decode(), job_id),
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



def test_training_target_is_hidden_while_queued_and_bound_when_starting(
    tmp_path,
) -> None:
    run_id, model_id, checkpoint_id = (str(uuid.uuid4()) for _ in range(3))

    class Planner:
        def resolve(self, request):
            return SimpleNamespace(
                operation="tiny_train",
                reservation={
                    "byte_count": 1_000_000,
                    "artifact_rows": 16,
                    "dataset_rows": 0,
                    "run_rows": 1,
                    "model_rows": 1,
                    "checkpoint_rows": 2,
                },
                resolved={
                    "request": dict(request),
                    "requested_final_step": request["steps"],
                    "_inputs": [],
                },
                run_id=run_id,
                model_id=model_id,
                tokenizer_id=None,
                checkpoint_ids=(checkpoint_id,),
            )

    class PersistedContexts:
        def __init__(self):
            self.values = {}

        def create_job_context_in(
            self, _connection, *, job_id, request_sha256, plan
        ):
            self.values[job_id] = (request_sha256, plan)

        def get_job_context(self, job_id):
            return SimpleNamespace(resolved=self.values[job_id][1].resolved)

        def materialize_worker_snapshot(self, job_id):
            request_sha256, plan = self.values[job_id]
            value = {
                "format": "llm-foundations-worker-input-v1",
                "job_id": job_id,
                "operation": plan.operation,
                "request_sha256": request_sha256,
                "runtime_profile": "wsl-cpu",
                "device": "cpu",
                "dependency_lock_sha256": "b" * 64,
                "companion_source_revision": "c" * 40,
                "ids": {
                    "run_id": plan.run_id,
                    "model_id": plan.model_id,
                    "tokenizer_id": plan.tokenizer_id,
                    "checkpoint_ids": list(plan.checkpoint_ids),
                },
                "resolved": {
                    key: value
                    for key, value in plan.resolved.items()
                    if key != "_inputs"
                },
                "inputs": [],
            }
            return SimpleNamespace(
                value=value,
                sha256=hashlib.sha256(canonical_json(value)).hexdigest(),
            )

    db = Database(tmp_path / "storage")
    db.initialize()
    registry = Registry(db, db.root, INSTANCE_ID, b"c" * 32)
    contexts = PersistedContexts()
    worker = ScriptedWorker([], None, protocol_eof=False)
    scheduler = Scheduler(
        db,
        registry,
        db.root,
        INSTANCE_ID,
        INSTALLATION_ID,
        "wsl-cpu",
        "0.1.0",
        {"tiny_train": {"available": True}},
        admission_planner=Planner(),
        training_store=contexts,
        worker_controller=ScriptedController(worker),
        clock=Clock(),
    )
    try:
        key = str(uuid.uuid4())
        submission = scheduler.submit(tiny_train(steps=3), key)
        queued = submission.job
        assert queued["requested_final_step"] is None
        assert (
            validate_schema(
                {"$ref": "#/components/schemas/Job"},
                queued,
                document="openapi.json",
            )
            == queued
        )
        queued_event = scheduler.list_events(queued["job_id"]).items[0]
        assert queued_event["payload"]["requested_final_step"] is None
        assert scheduler.submit(tiny_train(steps=3), key).response_body == submission.response_body

        starting = scheduler.dispatch_once()
        assert starting["state"] == "starting"
        assert starting["requested_final_step"] == 3
        assert (
            validate_schema(
                {"$ref": "#/components/schemas/Job"},
                starting,
                document="openapi.json",
            )
            == starting
        )
        starting_event = scheduler.list_events(queued["job_id"]).items[-1]
        assert starting_event["payload"] == {
            "state": "starting",
            "step": None,
            "requested_final_step": 3,
        }
    finally:
        scheduler._clear_active()
        db.close()

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

    store = ScriptedOperationStore()
    scheduler.operation_store = store
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
    assert store.calls == []


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
    store = ScriptedOperationStore()
    scheduler.operation_store = store
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
    assert [call[0] for call in store.calls] == [
        "prepare_terminal",
        "finalize_terminal",
        "complete_terminal",
    ]


def test_ownership_unknown_finalizes_store_metadata_and_pauses_scheduling(runtime):
    _, _, scheduler = runtime
    worker = ScriptedWorker(
        [{"type": "ready"}],
        None,
        protocol_eof=False,
        cancel_error=BrokenPipeError(),
        terminate_error=OSError("ownership proof lost"),
    )
    store = ScriptedOperationStore()
    scheduler.worker_controller = ScriptedController(worker)
    scheduler.operation_store = store
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    scheduler.dispatch_once()
    assert scheduler.tick()["state"] == "running"

    interrupted = scheduler.cancel(accepted["job_id"])

    assert interrupted["state"] == "interrupted"
    assert interrupted["terminal_reason"] == "worker_ownership_unknown"
    assert interrupted["error"]["code"] == "WORKER_OWNERSHIP_UNKNOWN"
    assert scheduler._scheduling_paused is True
    assert [call[0] for call in store.calls] == [
        "prepare_terminal",
        "finalize_terminal",
        "complete_terminal",
    ]


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


def test_nontraining_s2_ready_persists_zero_completed_steps(runtime):
    _, _, scheduler = runtime
    worker = ScriptedWorker(
        [
            {"type": "ready"},
            {
                "type": "error",
                "error": {
                    "code": "WORKER_PROTOCOL_ERROR",
                    "message": "Stopped after readiness.",
                    "retryable": False,
                    "field_errors": [],
                },
            },
            None,
        ],
        2,
    )
    scheduler.worker_controller = ScriptedController(worker)
    accepted = scheduler.submit(generate(), str(uuid.uuid4())).job
    scheduler.dispatch_once()

    failed = scheduler.tick()

    assert failed["state"] == "failed"
    assert failed["step"] == 0


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


def test_checkpoint_ready_is_committed_with_event_before_worker_ack(runtime):
    _, _, scheduler = runtime
    digest = "d" * 64
    names = (
        "model.safetensors",
        "optimizer.safetensors",
        "rng.safetensors",
        "tokenizer.json",
        "config.json",
        "trainer_state.json",
        "manifest.json",
    )
    proposal = {
        "type": "checkpoint_ready",
        "staging_name": "checkpoint-temp",
        "manifest_sha256": digest,
        "files": [
            {"name": name, "size": 1, "sha256": digest} for name in names
        ],
    }
    worker = ScriptedWorker(
        [{"type": "ready"}, proposal],
        None,
        protocol_eof=False,
    )
    store = ScriptedOperationStore()
    scheduler.worker_controller = ScriptedController(worker)
    scheduler.operation_store = store
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    scheduler.dispatch_once()

    running = scheduler.tick()

    assert running["state"] == "running"
    assert running["checkpoint_boundary"] == {
        "checkpoint_id": CHECKPOINT_ID,
        "step": 0,
    }
    assert worker.acks[0]["type"] == "checkpoint_committed"
    events = scheduler.list_events(accepted["job_id"]).items
    assert events[-1]["event_type"] == "checkpoint_committed"
    assert events[-1]["payload"]["checkpoint_id"] == CHECKPOINT_ID
    assert store.calls[-2:] == [
        ("commit_checkpoint", digest),
        ("complete_checkpoint", digest),
    ]
    assert scheduler.cancel(accepted["job_id"])["state"] == "cancelling"
    worker._messages.append(
        [
            {
                "type": "interrupted",
                "reason_code": "user_cancelled",
                "checkpoint_id": CHECKPOINT_ID,
                "checkpoint_step": 0,
                "error": None,
            },
            None,
        ]
    )
    worker._exit_code = 0
    worker.protocol_eof = True
    assert scheduler.tick()["state"] == "interrupted"


def test_training_failure_retains_latest_progress_after_prior_checkpoint(runtime):
    db, _, scheduler = runtime
    digest = "d" * 64
    names = (
        "model.safetensors",
        "optimizer.safetensors",
        "rng.safetensors",
        "tokenizer.json",
        "config.json",
        "trainer_state.json",
        "manifest.json",
    )
    proposal = {
        "type": "checkpoint_ready",
        "staging_name": "checkpoint-temp",
        "manifest_sha256": digest,
        "files": [
            {"name": name, "size": 1, "sha256": digest} for name in names
        ],
    }
    worker = ScriptedWorker(
        [
            {"type": "ready"},
            proposal,
            {
                "type": "event",
                "event_type": "progress",
                "payload": {
                    "current": 1,
                    "total": 3,
                    "unit": "updates",
                    "message": "step 1",
                },
            },
            {
                "type": "error",
                "error": {
                    "code": "WORKER_PROTOCOL_ERROR",
                    "message": "Stopped between checkpoints.",
                    "retryable": False,
                    "field_errors": [],
                },
            },
            None,
        ],
        2,
    )
    store = ScriptedOperationStore()
    scheduler.worker_controller = ScriptedController(worker)
    scheduler.operation_store = store
    accepted = scheduler.submit(tiny_train(), str(uuid.uuid4())).job
    set_requested_final_step(db, accepted["job_id"], 3)
    scheduler.dispatch_once()

    failed = scheduler.tick()

    assert failed["state"] == "failed"
    assert failed["step"] == 1
    assert failed["checkpoint_boundary"] == {
        "checkpoint_id": CHECKPOINT_ID,
        "step": 0,
    }
    prepared = next(call for call in store.calls if call[0] == "prepare_terminal")
    assert prepared[-1] == failed["finished_at"]


def test_checkpoint_pending_cleanup_waits_for_transaction_commit(
    runtime, monkeypatch
):
    _, _, scheduler = runtime
    digest = "d" * 64
    names = (
        "model.safetensors",
        "optimizer.safetensors",
        "rng.safetensors",
        "tokenizer.json",
        "config.json",
        "trainer_state.json",
        "manifest.json",
    )
    proposal = {
        "type": "checkpoint_ready",
        "staging_name": "checkpoint-temp",
        "manifest_sha256": digest,
        "files": [
            {"name": name, "size": 1, "sha256": digest} for name in names
        ],
    }
    worker = ScriptedWorker(
        [{"type": "ready"}],
        None,
        protocol_eof=False,
    )
    store = ScriptedOperationStore()
    scheduler.worker_controller = ScriptedController(worker)
    scheduler.operation_store = store
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    scheduler.dispatch_once()
    assert scheduler.tick()["state"] == "running"
    save = scheduler._save

    def fail_save(*_args, **_kwargs):
        raise sqlite3.OperationalError("forced checkpoint rollback")

    monkeypatch.setattr(scheduler, "_save", fail_save)
    scheduler._commit_worker_checkpoint(accepted["job_id"], proposal)

    running = scheduler.get(accepted["job_id"])
    assert running["state"] == "running"
    assert running["checkpoint_boundary"] is None
    assert [call[0] for call in store.calls] == [
        "prepare_checkpoint",
        "commit_checkpoint",
    ]
    assert worker.acks[0]["type"] == "commit_rejected"

    monkeypatch.setattr(scheduler, "_save", save)
    scheduler._commit_worker_checkpoint(accepted["job_id"], proposal)

    committed = scheduler.get(accepted["job_id"])
    assert committed["checkpoint_boundary"] == {
        "checkpoint_id": CHECKPOINT_ID,
        "step": 0,
    }
    assert [call[0] for call in store.calls] == [
        "prepare_checkpoint",
        "commit_checkpoint",
        "prepare_checkpoint",
        "commit_checkpoint",
        "complete_checkpoint",
    ]
    assert worker.acks[-1]["type"] == "checkpoint_committed"
    scheduler._clear_active()


def test_checkpoint_cannot_regress_below_observed_training_progress(runtime):
    db, _, scheduler = runtime
    digest = "d" * 64
    names = (
        "model.safetensors",
        "optimizer.safetensors",
        "rng.safetensors",
        "tokenizer.json",
        "config.json",
        "trainer_state.json",
        "manifest.json",
    )
    proposal = {
        "type": "checkpoint_ready",
        "staging_name": "checkpoint-temp",
        "manifest_sha256": digest,
        "files": [
            {"name": name, "size": 1, "sha256": digest} for name in names
        ],
    }
    worker = ScriptedWorker(
        [
            {"type": "ready"},
            {
                "type": "event",
                "event_type": "progress",
                "payload": {
                    "current": 1,
                    "total": 3,
                    "unit": "updates",
                    "message": "step 1",
                },
            },
            proposal,
        ],
        None,
        protocol_eof=False,
    )
    store = ScriptedOperationStore()
    scheduler.worker_controller = ScriptedController(worker)
    scheduler.operation_store = store
    accepted = scheduler.submit(tiny_train(), str(uuid.uuid4())).job
    set_requested_final_step(db, accepted["job_id"], 3)
    scheduler.dispatch_once()

    failed = scheduler.tick()

    assert failed["state"] == "failed"
    assert failed["step"] == 1
    assert worker.terminated == 1
    assert not any(call[0] == "commit_checkpoint" for call in store.calls)


def test_resume_dispatch_seeds_verified_parent_boundary_without_checkpoint_event(
    runtime,
):
    _, _, scheduler = runtime
    parent_id = str(uuid.uuid4())
    request = {
        "operation": "tiny_resume",
        "checkpoint_id": parent_id,
        "additional_steps": 25,
    }
    request_sha256 = hashlib.sha256(canonical_json(request)).hexdigest()
    allocations = {
        "run_id": str(uuid.uuid4()),
        "model_id": str(uuid.uuid4()),
        "tokenizer_id": None,
        "checkpoint_ids": [str(uuid.uuid4())],
    }
    snapshot = {
        "format": "llm-foundations-worker-input-v1",
        "job_id": None,
        "operation": "tiny_resume",
        "request_sha256": request_sha256,
        "runtime_profile": "wsl-cpu",
        "device": "cpu",
        "dependency_lock_sha256": "b" * 64,
        "companion_source_revision": "c" * 40,
        "ids": allocations,
        "resolved": {
            "parent_checkpoint": {
                "checkpoint_id": parent_id,
                "step": 25,
            },
            "requested_final_step": 50,
        },
        "inputs": [],
    }

    class SnapshotStore:
        def get_job_context(self, _job_id):
            return SimpleNamespace(resolved=snapshot["resolved"])

        def materialize_worker_snapshot(self, job_id):
            value = {**snapshot, "job_id": job_id}
            return SimpleNamespace(
                value=value,
                sha256=hashlib.sha256(canonical_json(value)).hexdigest(),
            )

    worker = ScriptedWorker(
        [
            {"type": "ready"},
            {
                "type": "interrupted",
                "reason_code": "user_cancelled",
                "checkpoint_id": parent_id,
                "checkpoint_step": 25,
                "error": None,
            },
            None,
        ],
        0,
    )
    scheduler.capabilities["tiny_resume"] = {"available": True}
    scheduler.training_store = SnapshotStore()
    scheduler.worker_controller = ScriptedController(worker)
    accepted = scheduler.submit(request, str(uuid.uuid4())).job

    starting = scheduler.dispatch_once()

    assert starting["checkpoint_boundary"] == {
        "checkpoint_id": parent_id,
        "step": 25,
    }
    assert starting["step"] == 25
    assert all(
        event["event_type"] != "checkpoint_committed"
        for event in scheduler.list_events(accepted["job_id"]).items
    )
    interrupted = scheduler.tick()
    assert interrupted["state"] == "interrupted"
    assert interrupted["run_id"] == allocations["run_id"]
    assert interrupted["step"] == 25
    assert interrupted["checkpoint_boundary"] == {
        "checkpoint_id": parent_id,
        "step": 25,
    }
    assert all(
        event["event_type"] != "checkpoint_committed"
        for event in scheduler.list_events(accepted["job_id"]).items
    )


def test_artifact_prepare_then_result_finalize_is_one_terminal_transaction(runtime):
    _, _, scheduler = runtime
    digest = hashlib.sha256(b"{}").hexdigest()
    result = {"operation": "model_prepare", "model_id": DATASET_ID, "artifact_ids": []}
    worker = ScriptedWorker(
        [
            {"type": "ready"},
            {
                "type": "artifact_ready",
                "role": "run_result",
                "staging_name": "result.partial",
                "size_bytes": 2,
                "sha256": digest,
            },
            {"type": "result", "operation": "model_prepare", "result": result},
            None,
        ],
        0,
    )
    store = ScriptedOperationStore()
    scheduler.worker_controller = ScriptedController(worker)
    scheduler.operation_store = store
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    scheduler.dispatch_once()

    completed = scheduler.tick()

    assert completed["state"] == "completed"
    assert completed["result"] == result
    assert completed["run_id"] == DATASET_ID
    assert worker.acks == [
        {
            "type": "artifact_prepared",
            "role": "run_result",
            "artifact_id": PREVIEW_ARTIFACT_ID,
            "sha256": digest,
        }
    ]
    assert [call[0] for call in store.calls] == [
        "prepare_artifact",
        "commit_artifact",
        "prepare_terminal",
        "finalize_terminal",
        "complete_terminal",
    ]


def test_terminal_pending_cleanup_waits_for_transaction_commit(
    runtime, monkeypatch
):
    _, _, scheduler = runtime
    worker = ScriptedWorker(
        [{"type": "ready"}],
        None,
        protocol_eof=False,
    )
    store = ScriptedOperationStore()
    scheduler.worker_controller = ScriptedController(worker)
    scheduler.operation_store = store
    accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    scheduler.dispatch_once()
    assert scheduler.tick()["state"] == "running"
    terminal = {
        "type": "error",
        "error": {
            "code": "WORKER_PROTOCOL_ERROR",
            "message": "Stopped after readiness.",
            "retryable": False,
            "field_errors": [],
        },
    }
    release_capacity = scheduler.registry.release_capacity

    def fail_release(*_args, **_kwargs):
        raise sqlite3.OperationalError("forced terminal rollback")

    monkeypatch.setattr(scheduler.registry, "release_capacity", fail_release)
    with pytest.raises(sqlite3.OperationalError, match="forced terminal rollback"):
        scheduler._finalize_worker_message(accepted["job_id"], terminal)

    assert scheduler.get(accepted["job_id"])["state"] == "running"
    assert [call[0] for call in store.calls] == [
        "prepare_terminal",
        "finalize_terminal",
    ]

    monkeypatch.setattr(scheduler.registry, "release_capacity", release_capacity)
    scheduler._finalize_worker_message(accepted["job_id"], terminal)

    failed = scheduler.get(accepted["job_id"])
    assert failed["state"] == "failed"
    assert [call[0] for call in store.calls] == [
        "prepare_terminal",
        "finalize_terminal",
        "prepare_terminal",
        "finalize_terminal",
        "complete_terminal",
    ]
    assert store.calls[-1][2] == (PREVIEW_ARTIFACT_ID,)
    scheduler._clear_active()


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



def test_invalid_completed_store_output_fails_job_and_allows_future_dispatch(runtime):
    db, _, scheduler = runtime
    digest = "d" * 64
    names = (
        "model.safetensors",
        "optimizer.safetensors",
        "rng.safetensors",
        "tokenizer.json",
        "config.json",
        "trainer_state.json",
        "manifest.json",
    )
    checkpoint = {
        "type": "checkpoint_ready",
        "staging_name": "checkpoint-temp",
        "manifest_sha256": digest,
        "files": [
            {"name": name, "size": 1, "sha256": digest} for name in names
        ],
    }
    metric_digest = hashlib.sha256(b"{}\n").hexdigest()
    metrics = {
        "type": "artifact_ready",
        "role": "training_metrics",
        "staging_name": "metrics.jsonl",
        "size_bytes": 3,
        "sha256": metric_digest,
    }

    class InvalidCompletedStore(ScriptedOperationStore):
        def prepare_terminal(
            self, job_id, *, state, reason_code, finished_at, result=None
        ):
            if state == "completed":
                self.calls.append(
                    ("reject_terminal", job_id, state, reason_code, finished_at)
                )
                raise OperationStoreError(
                    "WORKER_PROTOCOL_ERROR", "Metric output violates its schema."
                )
            return super().prepare_terminal(
                job_id,
                state=state,
                reason_code=reason_code,
                finished_at=finished_at,
                result=result,
            )

    worker = ScriptedWorker(
        [
            {"type": "ready"},
            checkpoint,
            metrics,
            {
                "type": "result",
                "operation": "tiny_train",
                "result": {"operation": "tiny_train"},
            },
            None,
        ],
        0,
    )
    store = InvalidCompletedStore()
    scheduler.worker_controller = ScriptedController(worker)
    scheduler.operation_store = store
    first = scheduler.submit(tiny_train(), str(uuid.uuid4())).job
    second = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
    set_requested_final_step(db, first["job_id"], 3)
    scheduler.dispatch_once()

    failed = scheduler.tick()

    assert failed["state"] == "failed"
    assert failed["error"]["code"] == "WORKER_PROTOCOL_ERROR"
    assert failed["checkpoint_boundary"] == {
        "checkpoint_id": CHECKPOINT_ID,
        "step": 0,
    }
    assert worker.terminated == 2
    assert worker.closed == 1
    assert [ack["type"] for ack in worker.acks] == [
        "checkpoint_committed",
        "artifact_prepared",
    ]
    assert [call[0] for call in store.calls] == [
        "prepare_checkpoint",
        "commit_checkpoint",
        "complete_checkpoint",
        "prepare_artifact",
        "commit_artifact",
        "reject_terminal",
        "prepare_terminal",
        "finalize_terminal",
        "complete_terminal",
    ]

    next_worker = ScriptedWorker([], None, protocol_eof=False)
    scheduler.worker_controller = ScriptedController(next_worker)
    dispatched = scheduler.dispatch_once()
    assert dispatched["job_id"] == second["job_id"]
    assert dispatched["state"] == "starting"
    scheduler._clear_active()


def _training_progress(current, total):
    return {
        "type": "event",
        "event_type": "progress",
        "payload": {
            "current": current,
            "total": total,
            "unit": "updates",
            "message": "Training",
        },
    }


def _persisted_progress(scheduler, job_id):
    return [
        event
        for event in scheduler.list_events(job_id, 0, 100).items
        if event["event_type"] == "progress"
    ]


def test_training_progress_updates_each_step_but_persists_only_sixteenth_milestones(
    runtime,
):
    db, _, scheduler = runtime
    worker = ScriptedWorker(
        [
            {"type": "ready"},
            *[_training_progress(step, 50) for step in range(1, 51)],
            {
                "type": "error",
                "error": {
                    "code": "WORKER_PROTOCOL_ERROR",
                    "message": "Stopped after all observed updates.",
                    "retryable": False,
                    "field_errors": [],
                },
            },
            None,
        ],
        2,
    )
    scheduler.worker_controller = ScriptedController(worker)
    scheduler.operation_store = ScriptedOperationStore()
    accepted = scheduler.submit(tiny_train(steps=50), str(uuid.uuid4())).job
    set_requested_final_step(db, accepted["job_id"], 50)
    scheduler.dispatch_once()

    failed = scheduler.tick()

    assert failed["state"] == "failed"
    assert failed["step"] == 50
    assert failed["progress"]["current"] == 50
    assert [event["payload"]["current"] for event in _persisted_progress(
        scheduler, accepted["job_id"]
    )] == [4, 7, 10, 13, 16, 19, 22, 25, 29, 32, 35, 38, 41, 44, 47, 50]


def test_training_progress_between_milestones_updates_job_without_inventing_event(
    runtime,
):
    db, _, scheduler = runtime
    worker = ScriptedWorker(
        [
            {"type": "ready"},
            _training_progress(1, 50),
            _training_progress(2, 50),
            _training_progress(3, 50),
        ],
        None,
        protocol_eof=False,
    )
    scheduler.worker_controller = ScriptedController(worker)
    accepted = scheduler.submit(tiny_train(steps=50), str(uuid.uuid4())).job
    set_requested_final_step(db, accepted["job_id"], 50)
    scheduler.dispatch_once()

    running = scheduler.tick()
    scheduler._clear_active()

    assert running["state"] == "running"
    assert running["step"] == 3
    assert running["progress"]["current"] == 3
    assert _persisted_progress(scheduler, accepted["job_id"]) == []


def test_training_failure_between_checkpoints_retains_latest_observed_update(runtime):
    db, _, scheduler = runtime
    digest = "d" * 64
    names = (
        "model.safetensors",
        "optimizer.safetensors",
        "rng.safetensors",
        "tokenizer.json",
        "config.json",
        "trainer_state.json",
        "manifest.json",
    )
    checkpoint = {
        "type": "checkpoint_ready",
        "staging_name": "checkpoint-temp",
        "manifest_sha256": digest,
        "files": [{"name": name, "size": 1, "sha256": digest} for name in names],
    }
    worker = ScriptedWorker(
        [
            {"type": "ready"},
            checkpoint,
            *[_training_progress(step, 50) for step in range(1, 6)],
            {
                "type": "error",
                "error": {
                    "code": "WORKER_PROTOCOL_ERROR",
                    "message": "Stopped between checkpoints.",
                    "retryable": False,
                    "field_errors": [],
                },
            },
            None,
        ],
        2,
    )
    scheduler.worker_controller = ScriptedController(worker)
    scheduler.operation_store = ScriptedOperationStore()
    accepted = scheduler.submit(tiny_train(steps=50), str(uuid.uuid4())).job
    set_requested_final_step(db, accepted["job_id"], 50)
    scheduler.dispatch_once()

    failed = scheduler.tick()

    assert failed["state"] == "failed"
    assert failed["step"] == 5
    assert failed["checkpoint_boundary"] == {
        "checkpoint_id": CHECKPOINT_ID,
        "step": 0,
    }
    assert [event["payload"]["current"] for event in _persisted_progress(
        scheduler, accepted["job_id"]
    )] == [4]


def test_resumed_training_persists_only_remaining_absolute_milestones(runtime):
    _, _, scheduler = runtime
    parent_id = str(uuid.uuid4())
    request = {
        "operation": "tiny_resume",
        "checkpoint_id": parent_id,
        "additional_steps": 25,
    }
    request_sha256 = hashlib.sha256(canonical_json(request)).hexdigest()
    allocations = {
        "run_id": str(uuid.uuid4()),
        "model_id": str(uuid.uuid4()),
        "tokenizer_id": None,
        "checkpoint_ids": [str(uuid.uuid4())],
    }
    resolved = {
        "parent_checkpoint": {"checkpoint_id": parent_id, "step": 25},
        "requested_final_step": 50,
    }

    class ResumeSnapshotStore:
        def get_job_context(self, _job_id):
            return SimpleNamespace(resolved=resolved)

        def materialize_worker_snapshot(self, job_id):
            value = {
                "format": "llm-foundations-worker-input-v1",
                "job_id": job_id,
                "operation": "tiny_resume",
                "request_sha256": request_sha256,
                "runtime_profile": "wsl-cpu",
                "device": "cpu",
                "dependency_lock_sha256": "b" * 64,
                "companion_source_revision": "c" * 40,
                "ids": allocations,
                "resolved": resolved,
                "inputs": [],
            }
            return SimpleNamespace(
                value=value,
                sha256=hashlib.sha256(canonical_json(value)).hexdigest(),
            )

    worker = ScriptedWorker(
        [
            {"type": "ready"},
            *[_training_progress(step, 50) for step in range(26, 51)],
            {
                "type": "error",
                "error": {
                    "code": "WORKER_PROTOCOL_ERROR",
                    "message": "Stopped after resumed updates.",
                    "retryable": False,
                    "field_errors": [],
                },
            },
            None,
        ],
        2,
    )
    scheduler.capabilities["tiny_resume"] = {"available": True}
    scheduler.training_store = ResumeSnapshotStore()
    scheduler.worker_controller = ScriptedController(worker)
    scheduler.operation_store = ScriptedOperationStore()
    accepted = scheduler.submit(request, str(uuid.uuid4())).job
    scheduler.dispatch_once()

    failed = scheduler.tick()

    assert failed["state"] == "failed"
    assert failed["step"] == 50
    assert [event["payload"]["current"] for event in _persisted_progress(
        scheduler, accepted["job_id"]
    )] == [29, 32, 35, 38, 41, 44, 47, 50]


def test_unknown_total_progress_persists_at_most_once_per_five_seconds_and_sixteen(
    runtime,
):
    _, _, scheduler = runtime

    class ManualClock:
        def __init__(self):
            self.value = datetime(2026, 10, 1, 13, 0, tzinfo=timezone.utc)

        def __call__(self):
            return self.value

        def advance(self, seconds):
            self.value += timedelta(seconds=seconds)

    worker = ScriptedWorker([{"type": "ready"}], None, protocol_eof=False)
    scheduler.worker_controller = ScriptedController(worker)
    accepted = scheduler.submit(generate(), str(uuid.uuid4())).job
    scheduler.dispatch_once()
    assert scheduler.tick()["state"] == "running"
    clock = ManualClock()
    scheduler._clock = clock

    def observe(current):
        scheduler._handle_worker_message(
            {
                "type": "event",
                "event_type": "progress",
                "payload": {
                    "current": current,
                    "total": None,
                    "unit": "records",
                    "message": "Working",
                },
            }
        )

    try:
        observe(1)
        clock.advance(4)
        observe(2)
        clock.advance(1)
        observe(3)
        for current in range(4, 23):
            clock.advance(5)
            observe(current)
    finally:
        scheduler._clear_active()

    running = scheduler.get(accepted["job_id"])
    events = _persisted_progress(scheduler, accepted["job_id"])
    assert running["progress"]["current"] == 22
    assert len(events) == 16
    assert events[0]["payload"]["current"] == 1
    assert events[1]["payload"]["current"] == 3
    timestamps = [datetime.fromisoformat(event["occurred_at"].replace("Z", "+00:00")) for event in events]
    assert all(
        later - earlier >= timedelta(seconds=5)
        for earlier, later in zip(timestamps, timestamps[1:])
    )
    scheduler._clear_active()
