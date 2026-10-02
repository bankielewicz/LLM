"""Regressions for backward wall-clock movement across scheduler mutations."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import json
import uuid

from llm_foundations_companion.database import Database
from llm_foundations_companion.registry import Registry
from llm_foundations_companion.scheduler import Scheduler
from llm_foundations_companion.schema import canonical_json, validate_schema
from test_scheduler import (
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


CREATED = "2026-10-02T01:04:01.679Z"
STARTED = "2026-10-02T01:04:01.715Z"
ROLLED_BACK = "2026-10-02T01:04:01.178Z"
EARLIER = "2026-10-02T01:04:01.100Z"
EARLIEST = "2026-10-02T01:04:01.050Z"
HOLD_DEADLINE = "2026-10-02T01:14:01.715Z"
PERSISTED_FLOOR = "2026-10-02T01:04:02.300Z"


class MutableClock:
    def __init__(self, value: str) -> None:
        self.set(value)

    def set(self, value: str) -> None:
        self.value = datetime.fromisoformat(value.replace("Z", "+00:00"))

    def __call__(self) -> datetime:
        return self.value


def _scheduler(tmp_path, clock: MutableClock, *, worker=None, store=None):
    database = Database(tmp_path / "storage")
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
        {"model_prepare": {"available": True}, "tiny_train": {"available": True}},
        admission_planner=TinyReservation(),
        operation_store=store,
        worker_controller=ScriptedController(worker) if worker is not None else None,
        clock=clock,
    )
    return database, registry, scheduler


def _job_sql(database: Database, job_id: str) -> dict[str, object]:
    with database.read() as connection:
        row = connection.execute(
            "SELECT created_at, updated_at, cancel_requested_at, record_json "
            "FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    return {
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "cancel_requested_at": row["cancel_requested_at"],
        "record": json.loads(row["record_json"]),
    }


def _assert_valid_job(job: dict[str, object]) -> None:
    assert validate_schema(
        {"$ref": "#/components/schemas/Job"}, job, document="openapi.json"
    ) == job


def _event_times(database: Database, job_id: str) -> list[str]:
    with database.read() as connection:
        return [
            row[0]
            for row in connection.execute(
                "SELECT occurred_at FROM job_events WHERE job_id = ? ORDER BY cursor",
                (job_id,),
            )
        ]


def test_ready_transition_clamps_observed_clock_reversal(tmp_path) -> None:
    clock = MutableClock(CREATED)
    worker = ScriptedWorker([{"type": "ready"}], None, protocol_eof=False)
    database, _, scheduler = _scheduler(tmp_path, clock, worker=worker)
    try:
        accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
        assert accepted["created_at"] == CREATED

        clock.set(STARTED)
        starting = scheduler.dispatch_once()
        assert starting["state"] == "starting"
        assert starting["started_at"] == STARTED

        clock.set(ROLLED_BACK)
        running = scheduler.tick()
        stored = _job_sql(database, accepted["job_id"])

        assert running["state"] == "running"
        assert running["updated_at"] == STARTED
        assert stored["updated_at"] == STARTED
        assert stored["record"]["updated_at"] == STARTED
        assert _event_times(database, accepted["job_id"]) == [
            CREATED,
            STARTED,
            STARTED,
        ]
        _assert_valid_job(running)
    finally:
        scheduler._clear_active()
        database.close()


def test_events_hold_and_terminal_share_persisted_floor(tmp_path) -> None:
    clock = MutableClock(CREATED)
    worker = ScriptedWorker([], None, protocol_eof=False)
    worker._messages = deque(
        [
            [{"type": "ready"}],
            [
                {
                    "type": "event",
                    "event_type": "phase_changed",
                    "payload": {"phase": "cancellable_hold"},
                },
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
            ],
            [
                {
                    "type": "error",
                    "error": {
                        "code": "WORKER_PROTOCOL_ERROR",
                        "message": "Stopped after rollback.",
                        "retryable": False,
                        "field_errors": [],
                    },
                }
            ],
        ]
    )
    store = ScriptedOperationStore()
    database, _, scheduler = _scheduler(
        tmp_path, clock, worker=worker, store=store
    )
    try:
        accepted = scheduler.submit(tiny_train(), str(uuid.uuid4())).job
        set_requested_final_step(database, accepted["job_id"], 3)
        clock.set(STARTED)
        scheduler.dispatch_once()

        clock.set(ROLLED_BACK)
        assert scheduler.tick()["state"] == "running"

        clock.set(EARLIER)
        held = scheduler.tick()
        assert held["phase"] == "cancellable_hold"
        assert held["hold_deadline_at"] == HOLD_DEADLINE
        assert held["updated_at"] == STARTED

        clock.set(EARLIEST)
        worker._exit_code = 2
        worker.protocol_eof = True
        failed = scheduler.tick()
        stored = _job_sql(database, accepted["job_id"])
        prepared = next(call for call in store.calls if call[0] == "prepare_terminal")
        times = _event_times(database, accepted["job_id"])

        assert failed["state"] == "failed"
        assert failed["started_at"] == STARTED
        assert failed["finished_at"] == STARTED
        assert failed["updated_at"] == STARTED
        assert prepared[-1] == failed["finished_at"]
        assert times == sorted(times)
        assert times[-1] == failed["finished_at"]
        assert stored["updated_at"] == failed["updated_at"]
        assert stored["record"]["updated_at"] == failed["updated_at"]
        _assert_valid_job(failed)
    finally:
        scheduler._clear_active()
        database.close()


def test_queued_cancellation_clamps_to_created_time(tmp_path) -> None:
    clock = MutableClock(CREATED)
    database, _, scheduler = _scheduler(tmp_path, clock)
    try:
        accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
        clock.set(ROLLED_BACK)
        interrupted = scheduler.cancel(accepted["job_id"])
        stored = _job_sql(database, accepted["job_id"])
        times = _event_times(database, accepted["job_id"])

        assert interrupted["state"] == "interrupted"
        assert interrupted["finished_at"] == CREATED
        assert interrupted["updated_at"] == CREATED
        assert times == sorted(times)
        assert times[-1] == CREATED
        assert stored["updated_at"] == CREATED
        assert stored["record"]["updated_at"] == CREATED
        _assert_valid_job(interrupted)
    finally:
        database.close()


def test_active_cancellation_persists_the_floored_request_time(tmp_path) -> None:
    clock = MutableClock(CREATED)
    worker = ScriptedWorker([{"type": "ready"}], None, protocol_eof=False)
    database, _, scheduler = _scheduler(tmp_path, clock, worker=worker)
    try:
        accepted = scheduler.submit(model_prepare(), str(uuid.uuid4())).job
        clock.set(STARTED)
        scheduler.dispatch_once()
        clock.set(ROLLED_BACK)
        assert scheduler.tick()["state"] == "running"

        clock.set(EARLIER)
        cancelling = scheduler.cancel(accepted["job_id"])
        stored = _job_sql(database, accepted["job_id"])
        times = _event_times(database, accepted["job_id"])

        assert cancelling["state"] == "cancelling"
        assert cancelling["updated_at"] == STARTED
        assert stored["updated_at"] == STARTED
        assert stored["cancel_requested_at"] == STARTED
        assert stored["record"]["updated_at"] == STARTED
        assert times == sorted(times)
        assert times[-1] == STARTED
        assert worker.cancelled == ["user_cancelled"]
        _assert_valid_job(cancelling)
    finally:
        scheduler._clear_active()
        database.close()


def test_restart_recovery_uses_sql_and_event_floor_when_json_is_stale(
    tmp_path,
) -> None:
    clock = MutableClock(CREATED)
    database, registry, original = _scheduler(tmp_path, clock)
    accepted = original.submit(model_prepare(), str(uuid.uuid4())).job
    job_id = accepted["job_id"]
    stale_record_time = "2026-10-02T01:04:01.700Z"
    cancel_time = "2026-10-02T01:04:02.200Z"
    with database.transaction() as connection:
        row = connection.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        record = json.loads(row["record_json"])
        record.update(
            state="cancelling",
            started_at=stale_record_time,
            updated_at=stale_record_time,
            queue_position=None,
        )
        event = original.events.append(
            connection,
            job_id=job_id,
            event_type="state_changed",
            payload={
                "state": "cancelling",
                "step": None,
                "requested_final_step": None,
            },
            occurred_at=PERSISTED_FLOOR,
        )
        record["last_cursor"] = event["cursor"]
        connection.execute(
            "UPDATE jobs SET state = 'cancelling', updated_at = ?, "
            "cancel_requested_at = ?, cancel_reason = 'user_cancelled', "
            "record_json = ? WHERE job_id = ?",
            (
                PERSISTED_FLOOR,
                cancel_time,
                canonical_json(record).decode("utf-8"),
                job_id,
            ),
        )

    clock.set(EARLIEST)
    store = ScriptedOperationStore()
    restarted = Scheduler(
        database,
        registry,
        database.root,
        INSTANCE_ID,
        INSTALLATION_ID,
        "wsl-cpu",
        "0.1.0",
        {"model_prepare": {"available": True}},
        admission_planner=TinyReservation(),
        operation_store=store,
        clock=clock,
    )
    try:
        assert restarted.recover() == 1
        interrupted = restarted.get(job_id)
        stored = _job_sql(database, job_id)
        prepared = next(call for call in store.calls if call[0] == "prepare_terminal")
        times = _event_times(database, job_id)

        assert interrupted["state"] == "interrupted"
        assert interrupted["finished_at"] == PERSISTED_FLOOR
        assert interrupted["updated_at"] == PERSISTED_FLOOR
        assert prepared[-1] == interrupted["finished_at"]
        assert times == sorted(times)
        assert times[-1] == interrupted["finished_at"]
        assert stored["updated_at"] == interrupted["updated_at"]
        assert stored["record"]["updated_at"] == interrupted["updated_at"]
        _assert_valid_job(interrupted)
    finally:
        database.close()
