"""Failure-path and response-safety regressions at the service boundary."""
import asyncio
import io
import json
import sqlite3
import uuid
from types import SimpleNamespace

import pytest

from llm_foundations_companion import api, cli, control, preflight
from llm_foundations_companion.errors import ApiError
from llm_foundations_companion.schema import canonical_json
from llm_foundations_companion.service import Service
from test_api import Harness
from test_service import inputs


def test_lifespan_acknowledges_start_and_stop_and_other_scopes_are_ignored():
    h = Harness()
    pending = iter([{'type': 'lifespan.startup'}, {'type': 'lifespan.shutdown'}])
    sent = []
    async def receive():
        return next(pending)
    async def send(message):
        sent.append(message)
    asyncio.run(h.app({'type': 'lifespan'}, receive, send))
    assert sent == [{'type': 'lifespan.startup.complete'}, {'type': 'lifespan.shutdown.complete'}]
    sent.clear()
    asyncio.run(h.app({'type': 'websocket'}, receive, send))
    assert sent == []


@pytest.mark.parametrize('headers', [
    {'content-length': '-1'}, {'content-length': '18446744073709551616'},
    {'content-length': '1,1'}, {'content-length': '+1'},
])
def test_bad_declared_lengths_reject_before_body(headers):
    h = Harness()
    status, body, _ = h.request('/api/v1/jobs', method='POST', body=b'{}',
        headers={'content-type': 'application/json', **headers})
    assert status == 400 and body['error']['reason_code'] == 'SCHEMA_INVALID'
    assert h.body_reads == 0


@pytest.mark.parametrize('headers', [
    {'content-encoding': 'gzip'}, {'content-type': 'application/json;charset=latin-1'},
])
def test_unsupported_json_encoding_rejects_before_body(headers):
    h = Harness()
    status, body, _ = h.request('/api/v1/jobs', method='POST', body=b'{}',
        headers={'content-type': 'application/json', **headers})
    assert status == 415 and body['error']['code'] == 'UNSUPPORTED_MEDIA_TYPE'
    assert h.body_reads == 0


def test_multipart_requires_length_before_reserving_storage():
    h = Harness()
    h.begin_upload = lambda *args: pytest.fail('length must be checked first')
    status, body, _ = h.request('/api/v1/datasets', method='POST',
        headers={'content-type': 'multipart/form-data; boundary=x'})
    assert status == 400 and body['error']['field_errors'][0]['field_path'] == 'Content-Length'
    assert h.body_reads == 0


def test_method_host_and_unexpected_body_checks():
    h = Harness()
    assert h.request('/', headers={'host': 'localhost:8765'})[0] == 403
    assert h.request('/', method='POST')[0] == 405
    assert h.request('/api/v1/runtime', method='POST')[0] == 405
    assert h.request('/api/v1/runtime', headers={'content-encoding': 'gzip'})[0] == 415
    status, body, _ = h.request('/api/v1/sessions/current', method='DELETE', body=b'x',
        headers={'content-length': '1'})
    assert status == 400 and body['error']['code'] == 'INVALID_REQUEST'
    assert h.request('/api/v1/sessions/current', method='DELETE',
        headers={'idempotency-key': None})[0] == 400


@pytest.mark.parametrize(('messages', 'length', 'code'), [
    ([{'type': 'http.disconnect'}], None, 'INVALID_REQUEST'),
    ([{'type': 'http.request', 'body': b'x'}], 2, 'VALIDATION_FAILED'),
    ([{'type': 'http.request', 'body': b'x' * (api.JSON_LIMIT+1)}], None, 'PAYLOAD_TOO_LARGE'),
])
def test_bodyless_route_rejects_disconnect_length_mismatch_and_overflow(messages, length, code):
    pending = iter(messages)
    async def receive():
        return next(pending)
    with pytest.raises(ApiError) as raised:
        asyncio.run(Harness().app.empty_body(receive, length))
    assert raised.value.code == code


@pytest.mark.parametrize(('media', 'policy', 'raw', 'inline'), [
    ('text/plain', 'text', b'hello', True),
    ('application/json', 'json', b'{"hello":1}', True),
    ('text/plain', 'text', b'\xff', False),
    ('text/html', 'text', b'<script>alert(1)</script>', False),
    ('text/plain', 'download_only', b'hello', False),
    ('text/plain', 'text', b'x' * 262145, False),
])
def test_artifact_content_never_renders_unsafe_media_inline(media, policy, raw, inline):
    h = Harness()
    stream = io.BytesIO(raw)
    artifact_id = str(uuid.uuid4())
    h.open_artifact = lambda identifier: ({'media_type': media, 'preview_policy': policy,
        'size_bytes': len(raw)}, stream)
    status, body, headers = h.request('/api/v1/artifacts/' + artifact_id + '/content')
    assert status == 200
    expected_body = json.loads(raw) if inline and media == 'application/json' else raw
    assert body == expected_body and stream.closed
    assert headers[b'content-type'] == (media.encode() if inline else b'application/octet-stream')
    assert headers[b'content-disposition'] == (b'inline' if inline else
        ('attachment; filename="artifact-' + artifact_id + '"').encode())
    assert headers[b'x-content-type-options'] == b'nosniff'


def test_artifact_range_rejects_without_opening_storage():
    h = Harness()
    h.open_artifact = lambda identifier: pytest.fail('ranges are unavailable')
    assert h.request('/api/v1/artifacts/' + str(uuid.uuid4()) + '/content',
        headers={'range': 'bytes=0-1'})[0] == 416


@pytest.mark.parametrize('failure', [ApiError('NOT_FOUND'), sqlite3.DatabaseError('private path'), OSError('private path')])
def test_artifact_midstream_failure_closes_once_without_json_or_second_headers(failure):
    class BrokenStream(io.BytesIO):
        def read(self, *args):
            raise failure
    h = Harness()
    stream = BrokenStream(b'x')
    h.open_artifact = lambda identifier: ({'media_type': 'application/octet-stream',
        'preview_policy': 'download_only', 'size_bytes': 1}, stream)
    status, body, _ = h.request('/api/v1/artifacts/' + str(uuid.uuid4()) + '/content')
    assert status == 200 and body == b'' and stream.closed
    assert sum(m['type'] == 'http.response.start' for m in h.messages) == 1


def test_persisted_response_must_be_canonical_and_valid_replay_is_marked():
    h = Harness()
    response = {'items': [], 'next_cursor': None}
    h.list_datasets = lambda params: json.dumps(response).encode()
    assert h.request('/api/v1/datasets')[0] == 500
    h.list_datasets = lambda params: canonical_json(response)
    assert h.request('/api/v1/datasets')[0:2] == (200, response)
    h.revoke_session = lambda identity, key: (204, b'', True)
    assert h.request('/api/v1/sessions/current', method='DELETE')[2][b'idempotent-replayed'] == b'true'
    h.revoke_session = lambda identity, key: (204, b'not-empty', False)
    assert h.request('/api/v1/sessions/current', method='DELETE')[0] == 500


def test_upload_staging_failure_releases_reservation(tmp_path, monkeypatch):
    with Service(tmp_path/'root', *inputs()) as service:
        def fail_create(owner):
            raise OSError('disk unavailable')
        monkeypatch.setattr(service.registry, 'create_staging_dir', fail_create)
        with pytest.raises(OSError):
            service.begin_upload(100, '/api/v1/datasets')
        with service.db.read() as connection:
            rows = connection.execute('SELECT state, released_at FROM reservations').fetchall()
            assert len(rows) == 1 and rows[0]['state'] == 'released' and rows[0]['released_at']


def test_upload_cleanup_release_failure_enters_recovery(tmp_path, monkeypatch):
    with Service(tmp_path/'root', *inputs()) as service:
        owner, staging = service.begin_upload(100, '/api/v1/datasets')
        def fail_release(identifier):
            raise sqlite3.OperationalError('disk unavailable')
        monkeypatch.setattr(service.registry, 'release_capacity_now', fail_release)
        service.end_upload(owner, staging)
        assert not staging.exists() and not service.storage_writable


def test_staging_cleanup_cannot_delete_an_unowned_target(tmp_path):
    with Service(tmp_path/'root', *inputs()) as service:
        outside = tmp_path/'preserve'; outside.mkdir()
        (outside/'sentinel').write_text('keep')
        with pytest.raises(RuntimeError):
            service._remove_staging(outside)
        assert (outside/'sentinel').read_text() == 'keep'


def test_audit_write_failure_enters_recovery_without_leaking_exception(tmp_path, monkeypatch):
    with Service(tmp_path/'root', *inputs()) as service:
        def fail_audit(**kwargs):
            raise sqlite3.OperationalError('private path')
        monkeypatch.setattr(service.db, 'audit_mutation', fail_audit)
        service.audit_request(str(uuid.uuid4()), None, 'test', 'failed')
        assert not service.storage_writable
        service.audit_request(str(uuid.uuid4()), None, 'test', 'failed')


def test_receipt_registration_failure_keeps_service_readable(tmp_path, monkeypatch):
    from llm_foundations_companion.registry import Registry
    def fail_register(*args, **kwargs):
        raise OSError('disk unavailable')
    monkeypatch.setattr(Registry, 'register_stream', fail_register)
    with Service(tmp_path/'root', *inputs()) as service:
        assert not service.storage_writable
        assert service.preflight_receipt['status'] == 'fail'
        assert service.preflight_receipt['receipt_artifact_id'] is None


def test_unexpected_control_start_failure_releases_root_lease(tmp_path, monkeypatch):
    original = control.ControlServer.start
    def fail_start(self):
        raise RuntimeError('test startup failure')
    monkeypatch.setattr(control.ControlServer, 'start', fail_start)
    with pytest.raises(RuntimeError):
        Service(tmp_path/'root', *inputs())
    monkeypatch.setattr(control.ControlServer, 'start', original)
    with Service(tmp_path/'root', *inputs()) as service:
        assert service.storage_writable
