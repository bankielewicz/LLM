"""HTTP boundary tests use actual ASGI messages, not a bypass route."""
import asyncio
import json
import uuid

import pytest

from llm_foundations_companion.api import Application, CSP
from llm_foundations_companion.auth import AuthManager
from llm_foundations_companion.errors import ApiError
from llm_foundations_companion.schema import load_document

ORIGIN = 'http://127.0.0.1:8765'


class Harness:
    def __init__(self):
        self.auth = AuthManager(str(uuid.uuid4()))
        grant = self.auth.issue_bootstrap()
        self.credentials = self.auth.exchange_bootstrap(grant.secret)
        self.app = Application(self)
        self.audits = []
        self.body_reads = 0

    def audit_request(self, *args):
        self.audits.append(args)

    def runtime_info(self):
        schema = load_document('openapi.json')['components']['schemas']['RuntimeInfo']
        return {'mode': 'local', 'api_version': '1.0', 'runtime_version': '0.1.0', 'instance_id': self.auth.instance_id, 'profile': 'wsl-cpu', 'device': 'cpu', 'storage_root_display': 'test-root', 'storage_writable': True, 'server_time': '2026-10-01T12:00:00.000Z', 'limits': {k: v['const'] for k, v in schema['properties']['limits']['properties'].items()}, 'capabilities': {}, 'gpu_offer': {'status': 'not_detected', 'gpu': None, 'cuda_profile': 'wsl-cuda', 'reason_code': None, 'cuda_environment_installed': False, 'explicit_profile': False}}

    def revoke_session(self, identity, key):
        self.auth.revoke_identity(identity)
        return 204, b'', False

    def list_events(self, job_id, cursor, limit):
        raise ApiError('NOT_FOUND')

    def request(self, path='/api/v1/runtime', method='GET', body=b'', query=b'', headers=None, authenticated=True):
        base = {'host': '127.0.0.1:8765', 'sec-fetch-site': 'none'}
        if authenticated:
            base['authorization'] = 'Bearer ' + self.credentials.access_token
        if method in {'POST', 'PUT', 'PATCH', 'DELETE'}:
            base.update({'origin': ORIGIN, 'x-llmf-csrf': self.credentials.csrf_token, 'idempotency-key': str(uuid.uuid4())})
        base.update(headers or {})
        base = {key: value for key, value in base.items() if value is not None}
        messages = []
        async def receive():
            self.body_reads += 1
            return {'type': 'http.request', 'body': body, 'more_body': False}
        async def send(message):
            messages.append(message)
        asyncio.run(self.app({'type': 'http', 'method': method, 'path': path, 'headers': [(k.encode(), v.encode()) for k, v in base.items()], 'query_string': query}, receive, send))
        self.messages = messages
        start = messages[0]
        raw = b''.join(m.get('body', b'') for m in messages[1:])
        response = json.loads(raw) if raw and dict(start['headers']).get(b'content-type') == b'application/json' else raw
        return start['status'], response, dict(start['headers'])


def test_health_contains_no_runtime_data():
    h = Harness()
    status, body, _ = h.request('/healthz', authenticated=False, headers={'host': 'localhost'})
    assert status == 200 and body == {'status': 'ok', 'api_version': '1.0'}


@pytest.mark.parametrize(('headers', 'method', 'code', 'reason'), [
    ({'host': 'localhost:8765', 'origin': 'https://evil.test'}, 'OPTIONS', 'ORIGIN_REJECTED', 'HOST_MISMATCH'),
    ({'origin': 'https://evil.test'}, 'OPTIONS', 'ORIGIN_REJECTED', 'PREFLIGHT_REJECTED'),
    ({'origin': 'https://evil.test', 'authorization': None}, 'POST', 'ORIGIN_REJECTED', 'ORIGIN_MISMATCH'),
    ({'origin': None}, 'POST', 'ORIGIN_REJECTED', 'ORIGIN_MISSING'),
    ({'sec-fetch-site': 'cross-site', 'authorization': None}, 'GET', 'ORIGIN_REJECTED', 'FETCH_SITE_REJECTED'),
    ({'authorization': None, 'x-llmf-csrf': None}, 'POST', 'AUTH_REQUIRED', None),
    ({'x-llmf-csrf': 'invalid', 'content-type': 'bad'}, 'POST', 'CSRF_REJECTED', None),
])
def test_authentication_order_does_not_read_body(headers, method, code, reason):
    h = Harness()
    status, body, response_headers = h.request('/api/v1/jobs', method=method, body=b'not JSON', headers=headers)
    assert status in {401, 403}
    assert body['error']['code'] == code
    assert body['error'].get('reason_code') == reason
    assert h.body_reads == 0
    assert not any(k.startswith(b'access-control') for k in response_headers)


def test_media_type_precedes_declared_size_and_body_parse():
    h = Harness()
    status, body, _ = h.request('/api/v1/jobs', method='POST', body=b'bad', headers={'content-type': 'text/plain', 'content-length': '2000000'})
    assert status == 415 and h.body_reads == 0
    status, body, _ = h.request('/api/v1/jobs', method='POST', body=b'bad', headers={'content-type': 'application/json', 'content-length': '2000000'})
    assert status == 413 and h.body_reads == 0


def test_valid_runtime_and_security_headers():
    status, body, headers = Harness().request()
    assert status == 200 and body['profile'] == 'wsl-cpu'
    assert headers[b'x-content-type-options'] == b'nosniff'
    assert headers[b'cache-control'] == b'no-store'
    uuid.UUID(headers[b'x-request-id'].decode())


def test_bootstrap_single_use_and_disconnect():
    h = Harness()
    grant = h.auth.issue_bootstrap()
    body = json.dumps({'bootstrap_secret': grant.secret}).encode()
    status, credentials, _ = h.request('/api/v1/sessions', method='POST', body=body, authenticated=False, headers={'content-type': 'application/json'})
    assert status == 201 and set(credentials) == {'access_token', 'csrf_token', 'instance_id', 'idle_expires_at', 'absolute_expires_at'}
    assert h.request('/api/v1/sessions', method='POST', body=body, authenticated=False, headers={'content-type': 'application/json'})[0] == 401
    assert h.request('/api/v1/sessions/current', method='DELETE')[0:2] == (204, b'')
    assert h.request()[0] == 401


@pytest.mark.parametrize('cursor', [b'-1', b'1.0', b'2147483648', b'x', b'%D9%A1', b'+1', b''])
@pytest.mark.parametrize('suffix', ['events', 'event-stream'])
def test_event_cursor_failure_is_json_before_stream(cursor, suffix):
    h = Harness()
    status, body, headers = h.request('/api/v1/jobs/' + str(uuid.uuid4()) + '/' + suffix, query=b'after_cursor=' + cursor)
    assert status == 400
    assert body['error']['reason_code'] == 'SCHEMA_INVALID'
    assert body['error']['field_errors'][0]['field_path'] == 'after_cursor'
    assert headers[b'content-type'] == b'application/json'


def test_unknown_and_repeated_query_parameters_reject():
    h = Harness()
    for query in (b'limit=1&limit=2', b'filename=/tmp/data', b'cursor=%FF'):
        status, body, _ = h.request('/api/v1/datasets', query=query)
        assert status == 400


def test_json_duplicate_keys_and_unknown_fields_reject():
    h = Harness()
    for body in (b'{"bootstrap_secret":"a","bootstrap_secret":"b"}', b'{"bootstrap_secret":"' + b'a' * 43 + b'","path":"/tmp/file"}'):
        status, data, _ = h.request('/api/v1/sessions', method='POST', body=body, authenticated=False, headers={'content-type': 'application/json'})
        assert status == 400


def test_static_path_traversal_and_unlisted_api_are_closed():
    h = Harness()
    assert h.request('/api/v1/not-a-route')[0] == 404
    for path in ('/../pyproject.toml', '/.git/config', '/x\\y', '/C:/secret'):
        assert h.request(path)[0] == 404


def test_reader_has_exact_csp_when_assets_bundled():
    status, body, headers = Harness().request('/')
    assert status == 200 and b'html' in body[:100].lower()
    assert headers[b'content-security-policy'].decode() == CSP


def test_sse_terminal_race_replays_committed_terminal_before_close():
    h = Harness()
    job_id = str(uuid.uuid4())
    event = {'job_id': job_id, 'cursor': 1, 'event_type': 'terminal', 'occurred_at': '2026-10-01T12:00:00.000Z', 'payload': {'state': 'interrupted', 'reason_code': 'worker_lost', 'checkpoint_id': None, 'checkpoint_step': None}}
    reads = []
    def events(job_id, cursor, limit):
        reads.append(cursor)
        return {'items': [] if len(reads)==1 else [event], 'next_after_cursor': cursor if len(reads)==1 else 1, 'has_more': False}
    h.list_events = events
    h.get_job = lambda job_id: terminal_job(job_id)
    status, raw, headers = h.request('/api/v1/jobs/' + job_id + '/event-stream', query=b'after_cursor=0')
    assert status == 200 and headers[b'content-type'] == b'text/event-stream'
    assert raw.startswith(b'id: 1\nevent: terminal\ndata: ')
    assert len(reads) == 2 and raw.count(b'event: terminal') == 1


def test_sse_reconnect_after_terminal_closes_without_duplicate_event():
    h = Harness()
    h.list_events = lambda *args: {'items': [], 'next_after_cursor': 1, 'has_more': False}
    h.get_job = lambda job_id: terminal_job(job_id)
    status, raw, _ = h.request('/api/v1/jobs/' + str(uuid.uuid4()) + '/event-stream', query=b'after_cursor=1')
    assert status == 200 and raw == b''


def terminal_job(job_id):
    from llm_foundations_companion.scheduler import Scheduler
    # A schema-valid cancelled queued job; use real initial record construction.
    job = Scheduler._initial_job(type('Initial', (), {'_progress': staticmethod(Scheduler._progress)})(), job_id=job_id, operation='tokenizer_train',
        now='2026-10-01T12:00:00.000Z', request_sha256='0'*64,
        artifact_id=str(uuid.uuid4()), idempotency_key=str(uuid.uuid4()), queue_position=1)
    job.update(state='interrupted', queue_position=None, finished_at=job['created_at'], terminal_reason='user_cancelled')
    return job


def generation_request():
    return {'operation': 'generate', 'prompt': 'hello', 'max_new_tokens': 1,
        'temperature': 1, 'top_p': 1, 'seed': 0,
        'preview_artifact_id': str(uuid.uuid4()), 'context_preview_digest': '0'*64,
        'checkpoint_id': str(uuid.uuid4())}


def test_api_preserves_schema_numeric_normalization_before_dispatch():
    h = Harness()
    received = []
    def submit(request, key):
        received.append(request)
        raise ApiError('CAPABILITY_UNAVAILABLE')
    h.submit_job = submit
    status, _, _ = h.request('/api/v1/jobs', method='POST',
        body=json.dumps(generation_request()).encode(), headers={'content-type': 'application/json'})
    assert status == 503 and len(received) == 1
    assert type(received[0]['temperature']) is float
    assert type(received[0]['top_p']) is float
    assert type(received[0]['seed']) is int


@pytest.mark.parametrize('result', [b'{broken', b'{"items":[],"next_cursor":null,"extra":1}', {'unexpected': True}])
def test_invalid_server_payload_is_internal_error(result):
    h = Harness()
    h.list_datasets = lambda params: result
    status, body, _ = h.request('/api/v1/datasets')
    assert status == 500 and body['error']['code'] == 'INTERNAL_ERROR'


def test_undeclared_server_status_is_internal_error():
    h = Harness()
    h.app.dispatch = lambda *args: (299, {}, False)
    status, body, _ = h.request()
    assert status == 500 and body['error']['code'] == 'INTERNAL_ERROR'


def test_malformed_initial_sse_page_is_json_before_headers():
    h = Harness()
    h.list_events = lambda *args: {'items': [{'event_type': 'bad\nevent: injected'}]}
    status, body, headers = h.request('/api/v1/jobs/' + str(uuid.uuid4()) + '/event-stream', query=b'after_cursor=0')
    assert status == 500 and body['error']['code'] == 'INTERNAL_ERROR'
    assert headers[b'content-type'] == b'application/json'


def test_sse_midstream_failure_closes_without_second_headers_or_error_payload():
    h = Harness()
    h.list_events = lambda *args: {'items': [], 'next_after_cursor': 0, 'has_more': False}
    calls = []
    def job(job_id):
        calls.append(job_id)
        if len(calls) > 1:
            raise OSError('private host details')
        return terminal_job(job_id)
    h.get_job = job
    status, raw, _ = h.request('/api/v1/jobs/' + str(uuid.uuid4()) + '/event-stream', query=b'after_cursor=0')
    assert status == 200 and raw == b''
    assert sum(m['type'] == 'http.response.start' for m in h.messages) == 1


def test_raw_sqlite_read_failure_has_bounded_storage_error():
    import sqlite3
    h = Harness()
    reasons = []
    h.db = type('Database', (), {'enter_read_only_recovery': lambda self, reason: reasons.append(reason)})()
    def failed_read(params):
        raise sqlite3.OperationalError('private root or corrupt database details')
    h.list_datasets = failed_read
    status, body, _ = h.request('/api/v1/datasets')
    assert status == 503 and body['error']['code'] == 'STORAGE_UNAVAILABLE'
    assert reasons == ['HTTP_STORAGE_FAILURE']
    assert 'private' not in json.dumps(body)
