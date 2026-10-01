"""Immutable per-job event log and replay pages."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from .errors import ApiError
from .schema import canonical_json
from .worker_protocol import WorkerProtocolError, validate_event_payload


MAX_EVENTS = 4_096
EVENT_LIMITS = {
    "state_changed": 6,
    "phase_changed": 32,
    "progress": 16,
    "metric": 2_001,
    "warning": 32,
    "checkpoint_committed": 2_001,
    "terminal": 1,
}


@dataclass(frozen=True)
class EventPage(Mapping[str, Any]):
    events: tuple[dict[str, Any], ...]
    next_after_cursor: int
    has_more: bool

    @property
    def items(self) -> tuple[dict[str, Any], ...]:
        return self.events

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": [dict(item) for item in self.events],
            "next_after_cursor": self.next_after_cursor,
            "has_more": self.has_more,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(("items", "next_after_cursor", "has_more"))

    def __len__(self) -> int:
        return 3


class EventLog:
    def __init__(self, database: Any) -> None:
        self.database = database

    def append(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        occurred_at: str,
    ) -> dict[str, Any]:
        try:
            validate_event_payload(event_type, payload)
        except WorkerProtocolError:
            raise
        row = connection.execute(
            """
            SELECT COUNT(*) AS total,
                   COALESCE(MAX(cursor), 0) AS last_cursor,
                   SUM(CASE WHEN event_type = ? THEN 1 ELSE 0 END) AS typed
            FROM job_events WHERE job_id = ?
            """,
            (event_type, job_id),
        ).fetchone()
        if row is None:
            raise WorkerProtocolError("event state is unavailable")
        total, last_cursor, typed = int(row[0]), int(row[1]), int(row[2] or 0)
        limit = EVENT_LIMITS[event_type]
        if typed >= limit or total >= MAX_EVENTS or (
            event_type != "terminal" and total >= MAX_EVENTS - 1
        ):
            raise WorkerProtocolError("job event cardinality exceeded")
        cursor = last_cursor + 1
        if cursor > 2_147_483_647:
            raise WorkerProtocolError("job cursor overflow")
        body = canonical_json(dict(payload)).decode("utf-8")
        connection.execute(
            """
            INSERT INTO job_events(job_id, cursor, event_type, occurred_at, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job_id, cursor, event_type, occurred_at, body),
        )
        return {
            "job_id": job_id,
            "cursor": cursor,
            "event_type": event_type,
            "occurred_at": occurred_at,
            "payload": dict(payload),
        }

    def list(self, job_id: str, after_cursor: int = 0, limit: int = 100) -> EventPage:
        if isinstance(after_cursor, bool) or not isinstance(after_cursor, int):
            raise ValueError("after_cursor must be an integer")
        if not 0 <= after_cursor <= 2_147_483_647:
            raise ValueError("after_cursor is outside its bounds")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("event page limit must be between 1 and 500")
        with self.database.read() as connection:
            job = connection.execute(
                "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if job is None:
                raise ApiError("NOT_FOUND")
            last = int(
                connection.execute(
                    "SELECT COALESCE(MAX(cursor), 0) FROM job_events WHERE job_id = ?",
                    (job_id,),
                ).fetchone()[0]
            )
            if after_cursor > last:
                raise ApiError(
                    "VALIDATION_FAILED",
                    reason_code="SEMANTIC_INVALID",
                    field_errors=[
                        {
                            "field_path": "after_cursor",
                            "message": "Cursor is beyond the job event log.",
                        }
                    ],
                )
            rows = tuple(
                connection.execute(
                    """
                    SELECT job_id, cursor, event_type, occurred_at, payload_json
                    FROM job_events
                    WHERE job_id = ? AND cursor > ?
                    ORDER BY cursor ASC LIMIT ?
                    """,
                    (job_id, after_cursor, limit + 1),
                )
            )
        has_more = len(rows) > limit
        rows = rows[:limit]
        items = tuple(
            {
                "job_id": str(row[0]),
                "cursor": int(row[1]),
                "event_type": str(row[2]),
                "occurred_at": str(row[3]),
                "payload": json.loads(row[4]),
            }
            for row in rows
        )
        next_cursor = int(items[-1]["cursor"]) if items else after_cursor
        return EventPage(items, next_cursor, has_more)


__all__ = ["EVENT_LIMITS", "EventLog", "EventPage", "MAX_EVENTS"]
