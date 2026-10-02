"""Event stream termination, reauthorization, pagination and heartbeat behavior."""
import asyncio
import uuid
from types import SimpleNamespace

import pytest

from llm_foundations_companion import api
from test_api import Harness, terminal_job


def fixture_stream():
    h = Harness()
    job_id = str(uuid.uuid4())
    running = terminal_job(job_id)
    running.update(state='running', started_at=running['created_at'],
                   finished_at=None, terminal_reason=None)
    h.get_job = lambda _: running
    h.list_events = lambda *args: {'items': [], 'next_after_cursor': 0, 'has_more': False}
    headers = {'authorization': 'Bearer ' + h.credentials.access_token,
               'host': '127.0.0.1:8765', 'sec-fetch-site': 'none'}
    return h, job_id, headers


def run_stream(h, job_id, headers, *, receive=None, send=None):
    messages = []
    state = {'receive_finalized': False}
    async def pending_receive():
        try:
            await asyncio.Event().wait()
        finally:
            state['receive_finalized'] = True
    async def collect(message):
        messages.append(message)
        if send is not None:
            await send(message)
    async def invoke():
        await h.app.event_stream({'id': job_id, 'after_cursor': 0}, headers,
            receive or pending_receive, collect, str(uuid.uuid4()))
        await asyncio.sleep(0)
    asyncio.run(invoke())
    return messages, state


def test_stream_page_limit_preserves_every_cursor_and_closes():
    h, job_id, headers = fixture_stream()
    events = [{'job_id': job_id, 'cursor': i, 'event_type': 'checkpoint_committed',
        'occurred_at': '2026-10-01T12:00:00.000Z',
        'payload': {'checkpoint_id': str(uuid.uuid4()), 'step': i, 'sha256': 'a'*64}}
        for i in range(1, 501)]
    calls = []
    def page(identifier, cursor, limit):
        calls.append((identifier, cursor, limit))
        return {'items': events, 'next_after_cursor': 500, 'has_more': True}
    h.list_events = page
    messages, _ = run_stream(h, job_id, headers)
    frames = [x['body'] for x in messages if x.get('body')]
    assert len(frames) == 500
    assert [int(x.split(b'\n')[0].split(b': ')[1]) for x in frames] == list(range(1, 501))
    assert calls == [(job_id, 0, 500)]
    assert messages[-1] == {'type': 'http.response.body', 'body': b''}
    assert len([x for x in messages if x['type'] == 'http.response.start']) == 1


def test_stream_revocation_between_pages_suppresses_fetched_event(monkeypatch):
    h, job_id, headers = fixture_stream()
    real_sleep = asyncio.sleep
    calls = []
    def page(identifier, cursor, limit):
        calls.append(cursor)
        event = {'job_id': job_id, 'cursor': 1, 'event_type': 'warning',
                 'occurred_at': '2026-10-01T12:00:00.000Z',
                 'payload': {'code': 'test', 'message': 'Must remain unseen after revocation'}}
        return {'items': [] if len(calls)==1 else [event],
                'next_after_cursor': 0 if len(calls)==1 else 1, 'has_more': False}
    h.list_events = page
    async def sleep(delay):
        if delay:
            identity = h.auth.authorize('GET', headers, refresh_idle=False)
            h.auth.revoke_identity(identity)
        await real_sleep(0)
    monkeypatch.setattr(api.asyncio, 'sleep', sleep)
    messages, state = run_stream(h, job_id, headers)
    assert calls == [0, 0]
    assert not any(x.get('body') for x in messages)
    assert state['receive_finalized']
    assert messages[-1] == {'type': 'http.response.body', 'body': b''}


def test_stream_disconnect_stops_without_post_disconnect_frames(monkeypatch):
    h, job_id, headers = fixture_stream()
    real_sleep = asyncio.sleep
    calls = []
    async def receive():
        calls.append('disconnect')
        return {'type': 'http.disconnect'}
    async def sleep(delay):
        await real_sleep(0)
    monkeypatch.setattr(api.asyncio, 'sleep', sleep)
    messages, _ = run_stream(h, job_id, headers, receive=receive)
    assert calls == ['disconnect']
    assert [x['type'] for x in messages] == ['http.response.start']


def test_stream_heartbeat_then_deadline_closes_and_cancels_receiver(monkeypatch):
    h, job_id, headers = fixture_stream()
    clock = [0.0]
    monkeypatch.setattr(api, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    real_sleep = asyncio.sleep
    async def sleep(delay):
        if delay:
            clock[0] += 20 if clock[0] == 0 else 300
        await real_sleep(0)
    monkeypatch.setattr(api.asyncio, 'sleep', sleep)
    messages, state = run_stream(h, job_id, headers)
    assert [x.get('body') for x in messages[1:]] == [b': heartbeat\n\n', b'']
    assert state['receive_finalized']


def test_stream_network_send_failure_cancels_receiver():
    h, job_id, headers = fixture_stream()
    job = terminal_job(job_id)
    h.get_job = lambda _: job
    finalized = []
    async def receive():
        try:
            await asyncio.Event().wait()
        finally:
            finalized.append(True)
    async def send(message):
        if message['type'] == 'http.response.body':
            await asyncio.sleep(0)
            raise OSError('connection closed')
    with pytest.raises(OSError, match='connection closed'):
        run_stream(h, job_id, headers, receive=receive, send=send)
    assert finalized == [True]
