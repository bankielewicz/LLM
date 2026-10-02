"""Immutable event log cardinality and cursor boundaries."""
import uuid

import pytest

from llm_foundations_companion.events import EventLog, EventPage
from llm_foundations_companion.errors import ApiError
from llm_foundations_companion.worker_protocol import WorkerProtocolError
from test_scheduler import runtime, model_prepare


@pytest.mark.parametrize(('cursor', 'limit'), [(True, 100), ('0', 100), (-1, 100),
    (2147483648, 100), (0, True), (0, '100'), (0, 0), (0, 501)])
def test_event_page_bounds_reject_before_database_access(cursor, limit):
    with pytest.raises(ValueError):
        EventLog(None).list(str(uuid.uuid4()), cursor, limit)


def test_event_missing_job_has_public_not_found(runtime):
    db, _, _ = runtime
    with pytest.raises(ApiError) as raised: EventLog(db).list(str(uuid.uuid4()))
    assert raised.value.code == 'NOT_FOUND'


def test_event_page_mapping_has_stable_public_keys_and_defensive_rows():
    row = {'cursor': 1, 'event_type': 'warning'}
    page = EventPage((row,), 1, False)
    assert dict(page) == {'items': [row], 'next_after_cursor': 1, 'has_more': False}
    assert len(page) == 3
    copy = page['items']; copy[0]['cursor'] = 99
    assert page.items[0]['cursor'] == 1


def test_event_typed_limit_rejects_without_partial_insert(runtime):
    db, _, scheduler = runtime
    job_id = scheduler.submit(model_prepare(), str(uuid.uuid4())).job['job_id']
    log = EventLog(db)
    for index in range(32):
        with db.transaction() as connection:
            log.append(connection, job_id=job_id, event_type='warning',
                payload={'code': 'TEST_WARNING', 'message': str(index)}, occurred_at='2026-10-01T12:00:00.000Z')
    with pytest.raises(WorkerProtocolError, match='cardinality'):
        with db.transaction() as connection:
            log.append(connection, job_id=job_id, event_type='warning',
                payload={'code': 'TEST_WARNING', 'message': 'over limit'}, occurred_at='2026-10-01T12:00:00.000Z')
    assert len(log.list(job_id, 0, 500).items) == 33


def test_event_cursor_overflow_and_invalid_payload_never_insert(runtime):
    db, _, scheduler = runtime
    job_id = scheduler.submit(model_prepare(), str(uuid.uuid4())).job['job_id']
    log = EventLog(db)
    with db.transaction() as connection:
        connection.execute('INSERT INTO job_events(job_id,cursor,event_type,occurred_at,payload_json) VALUES (?,?,?,?,?)',
            (job_id,2147483647,'warning','2026-10-01T12:00:00.000Z','{"code":"TEST_WARNING","message":"last cursor"}'))
    with pytest.raises(WorkerProtocolError, match='overflow'):
        with db.transaction() as connection:
            log.append(connection, job_id=job_id, event_type='warning',
                payload={'code':'TEST_WARNING','message':'overflow'}, occurred_at='2026-10-01T12:00:00.000Z')
    with pytest.raises(WorkerProtocolError):
        with db.transaction() as connection:
            log.append(connection, job_id=job_id, event_type='warning',
                payload={'unrecognized':True}, occurred_at='2026-10-01T12:00:00.000Z')
    assert len(log.list(job_id,0,500).items) == 2
