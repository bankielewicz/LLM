"""Installed-package supporting checks for resume and scheduler loss boundaries."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import unittest
import uuid

from llm_foundations_companion.database import Database
from llm_foundations_companion.registry import Registry
from llm_foundations_companion.scheduler import Scheduler
from llm_foundations_companion.schema import canonical_json
from llm_foundations_companion.worker_context import WorkerContext
from llm_foundations_companion.worker_protocol import CancellationToken


INSTANCE_ID = "123e4567-e89b-42d3-a456-426614174000"
INSTALLATION_ID = "123e4567-e89b-42d3-a456-426614174001"
PARENT_ID = "123e4567-e89b-42d3-a456-426614174003"


class TinyReservation:
    def plan(self, request: object) -> dict[str, int]:
        return {
            "byte_count": 1,
            "artifact_rows": 1,
            "dataset_rows": 0,
            "run_rows": 0,
            "model_rows": 0,
            "checkpoint_rows": 0,
        }


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        self.value += timedelta(milliseconds=1)
        return self.value


class ScriptedWorker:
    def __init__(
        self,
        messages: list[object],
        exit_code: int | None,
        *,
        protocol_eof: bool = True,
        cancel_error: BaseException | None = None,
    ) -> None:
        self.process = SimpleNamespace(pid=4242)
        self.spawn_nonce = "f" * 64
        self.protocol_eof = protocol_eof
        self._messages = deque((list(messages),))
        self._exit_code = exit_code
        self._cancel_error = cancel_error
        self.terminated = 0
        self.closed = 0
        self.cancelled: list[str] = []
        self.acks: list[dict[str, object]] = []

    def messages(self) -> list[object]:
        return self._messages.popleft() if self._messages else []

    def poll(self) -> int | None:
        return self._exit_code

    def request_cancel(self, reason: str) -> None:
        if self._cancel_error is not None:
            raise self._cancel_error
        self.cancelled.append(reason)

    def send_ack(self, message: dict[str, object]) -> None:
        self.acks.append(dict(message))

    def terminate_owned(self, expected_nonce: str) -> None:
        if expected_nonce != self.spawn_nonce:
            raise AssertionError("worker ownership nonce differs")
        self.terminated += 1

    def close(self) -> None:
        self.closed += 1


class ScriptedController:
    def __init__(self, worker: ScriptedWorker) -> None:
        self.worker = worker

    def launch(self, **values: object) -> ScriptedWorker:
        return self.worker


class ScriptedOperationStore:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def prepare_terminal(
        self,
        job_id: str,
        *,
        state: str,
        reason_code: str,
        finished_at: str,
        result: object = None,
    ) -> object:
        self.calls.append(
            ("prepare_terminal", job_id, state, reason_code, finished_at)
        )
        return SimpleNamespace(
            state=state,
            result=result,
            finished_at=finished_at,
            pending_artifact_ids=(),
        )

    def finalize_terminal_in(
        self, connection: sqlite3.Connection, prepared: object
    ) -> object:
        self.calls.append(("finalize_terminal", prepared.state))
        return SimpleNamespace(
            result=prepared.result,
            run_id=None,
            model_id=None,
            artifact_ids=(),
        )

    def complete_terminal(self, prepared: object) -> None:
        self.calls.append(
            ("complete_terminal", prepared.state, prepared.pending_artifact_ids)
        )


class SnapshotStore:
    def __init__(self, snapshot: dict[str, object]) -> None:
        self.snapshot = snapshot

    def get_job_context(self, job_id: str) -> object:
        return SimpleNamespace(
            job_id=job_id,
            resolved=dict(self.snapshot["resolved"]),
        )

    def materialize_worker_snapshot(self, job_id: str) -> object:
        value = {**self.snapshot, "job_id": job_id}
        return SimpleNamespace(
            value=value,
            sha256=hashlib.sha256(canonical_json(value)).hexdigest(),
        )


class InstalledSchedulerBoundaryChecks(unittest.TestCase):
    def setUp(self) -> None:
        parent = Path(os.environ["LLMF_S2_NATIVE_ARTIFACT_ROOT"])
        self.root = parent / "supporting-scheduler" / str(uuid.uuid4())
        self.database = Database(self.root)
        self.database.initialize()
        self.registry = Registry(
            self.database,
            self.database.root,
            INSTANCE_ID,
            b"s2-native-scheduler-cursor-key!!",
        )
        self.scheduler = Scheduler(
            self.database,
            self.registry,
            self.database.root,
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

    def tearDown(self) -> None:
        self.scheduler.shutdown()
        self.database.close()

    @staticmethod
    def model_prepare() -> dict[str, object]:
        return {
            "operation": "model_prepare",
            "model_profile_id": "smollm2-135m-instruct-v1",
            "accept_download": True,
        }

    def test_resume_dispatch_starts_at_verified_parent_boundary(self) -> None:
        request = {
            "operation": "tiny_resume",
            "checkpoint_id": PARENT_ID,
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
                    "checkpoint_id": PARENT_ID,
                    "step": 25,
                },
                "requested_final_step": 50,
            },
            "inputs": [],
        }
        worker = ScriptedWorker(
            [
                {"type": "ready"},
                {
                    "type": "interrupted",
                    "reason_code": "user_cancelled",
                    "checkpoint_id": PARENT_ID,
                    "checkpoint_step": 25,
                    "error": None,
                },
                None,
            ],
            0,
        )
        store = ScriptedOperationStore()
        self.scheduler.capabilities["tiny_resume"] = {"available": True}
        self.scheduler.training_store = SnapshotStore(snapshot)
        self.scheduler.worker_controller = ScriptedController(worker)
        self.scheduler.operation_store = store
        accepted = self.scheduler.submit(request, str(uuid.uuid4())).job

        starting = self.scheduler.dispatch_once()
        self.assertEqual(
            starting["checkpoint_boundary"],
            {"checkpoint_id": PARENT_ID, "step": 25},
        )
        self.assertEqual(starting["step"], 25)
        self.assertEqual(starting["requested_final_step"], 50)
        self.assertFalse(
            any(
                event["event_type"] == "checkpoint_committed"
                for event in self.scheduler.list_events(accepted["job_id"]).items
            )
        )
        interrupted = self.scheduler.tick()
        self.assertEqual(interrupted["state"], "interrupted")
        self.assertEqual(
            interrupted["checkpoint_boundary"],
            {"checkpoint_id": PARENT_ID, "step": 25},
        )
        self.assertEqual(
            [call[0] for call in store.calls],
            ["prepare_terminal", "finalize_terminal", "complete_terminal"],
        )
        self.assertFalse(
            any(
                event["event_type"] == "checkpoint_committed"
                for event in self.scheduler.list_events(accepted["job_id"]).items
            )
        )
        self.s2_observation = {
            "job_id": accepted["job_id"],
            "parent_checkpoint_id": PARENT_ID,
            "parent_step": 25,
            "checkpoint_committed_events": 0,
            "terminal_finalization": [call[0] for call in store.calls],
        }

    def test_started_protocol_loss_finalizes_terminal_metadata(self) -> None:
        worker = ScriptedWorker(
            [{"type": "ready"}],
            None,
            protocol_eof=False,
            cancel_error=BrokenPipeError(),
        )
        store = ScriptedOperationStore()
        self.scheduler.worker_controller = ScriptedController(worker)
        self.scheduler.operation_store = store
        accepted = self.scheduler.submit(
            self.model_prepare(), str(uuid.uuid4())
        ).job
        self.scheduler.dispatch_once()
        self.assertEqual(self.scheduler.tick()["state"], "running")

        interrupted = self.scheduler.cancel(accepted["job_id"])
        self.assertEqual(interrupted["state"], "interrupted")
        self.assertEqual(interrupted["terminal_reason"], "worker_lost")
        self.assertEqual(worker.terminated, 1)
        self.assertEqual(worker.closed, 1)
        self.assertEqual(
            [call[0] for call in store.calls],
            ["prepare_terminal", "finalize_terminal", "complete_terminal"],
        )
        self.s2_observation = {
            "job_id": accepted["job_id"],
            "terminal_reason": interrupted["terminal_reason"],
            "worker_terminated": worker.terminated,
            "worker_closed": worker.closed,
            "terminal_finalization": [call[0] for call in store.calls],
        }

    def test_recovery_finalizes_prior_active_job_and_preserves_fifo(self) -> None:
        store = ScriptedOperationStore()
        self.scheduler.operation_store = store
        active = self.scheduler.submit(
            self.model_prepare(), str(uuid.uuid4())
        ).job
        queued = self.scheduler.submit(
            self.model_prepare(), str(uuid.uuid4())
        ).job
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT record_json FROM jobs WHERE job_id = ?",
                (active["job_id"],),
            ).fetchone()
            record = json.loads(row[0])
            now = "2026-10-01T12:00:01.000Z"
            record.update(
                state="running",
                started_at=now,
                updated_at=now,
                queue_position=None,
            )
            connection.execute(
                "UPDATE jobs SET state='running', updated_at=?, record_json=? WHERE job_id=?",
                (now, canonical_json(record).decode(), active["job_id"]),
            )
            event = self.scheduler.events.append(
                connection,
                job_id=active["job_id"],
                event_type="state_changed",
                payload={
                    "state": "running",
                    "step": None,
                    "requested_final_step": None,
                },
                occurred_at=now,
            )
            record["last_cursor"] = event["cursor"]
            connection.execute(
                "UPDATE jobs SET record_json=? WHERE job_id=?",
                (canonical_json(record).decode(), active["job_id"]),
            )

        self.assertEqual(self.scheduler.recover(), 1)
        recovered = self.scheduler.get(active["job_id"])
        self.assertEqual(recovered["state"], "interrupted")
        self.assertEqual(recovered["terminal_reason"], "service_restarted")
        retained = self.scheduler.get(queued["job_id"])
        self.assertEqual(retained["state"], "queued")
        self.assertEqual(retained["queue_position"], 1)
        self.assertEqual(
            [call[0] for call in store.calls],
            ["prepare_terminal", "finalize_terminal", "complete_terminal"],
        )
        self.s2_observation = {
            "active_job_id": active["job_id"],
            "queued_job_id": queued["job_id"],
            "terminal_reason": recovered["terminal_reason"],
            "queued_position": retained["queue_position"],
            "terminal_finalization": [call[0] for call in store.calls],
        }

    def test_worker_context_starts_at_verified_parent_boundary(self) -> None:
        staging = self.root / "worker-context-staging"
        staging.mkdir()
        allocations = {
            "run_id": str(uuid.uuid4()),
            "model_id": str(uuid.uuid4()),
            "tokenizer_id": None,
            "checkpoint_ids": [str(uuid.uuid4())],
        }
        snapshot = {
            "format": "llm-foundations-worker-input-v1",
            "job_id": str(uuid.uuid4()),
            "operation": "tiny_resume",
            "request_sha256": "d" * 64,
            "runtime_profile": "wsl-cpu",
            "device": "cpu",
            "dependency_lock_sha256": "b" * 64,
            "companion_source_revision": "c" * 40,
            "ids": allocations,
            "resolved": {
                "parent_checkpoint": {
                    "checkpoint_id": PARENT_ID,
                    "step": 25,
                }
            },
            "inputs": [],
        }
        context = WorkerContext(
            job_id=str(snapshot["job_id"]),
            instance_id=INSTANCE_ID,
            runtime_profile="wsl-cpu",
            staging_path=staging,
            cancellation=CancellationToken(io.BytesIO()),
            protocol=io.BytesIO(),
            control=io.BytesIO(),
            input_snapshot=snapshot,
            output_allocations=allocations,
        )
        self.assertEqual(
            context.last_checkpoint,
            {"checkpoint_id": PARENT_ID, "step": 25},
        )
        self.s2_observation = {
            "parent_checkpoint_id": PARENT_ID,
            "parent_step": 25,
            "worker_last_checkpoint": dict(context.last_checkpoint),
        }


if __name__ == "__main__":
    unittest.main()
