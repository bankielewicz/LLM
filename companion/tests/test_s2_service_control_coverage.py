"""Owner IPC auditing and storage recovery at the service boundary."""
import json
import sqlite3
import uuid
from types import SimpleNamespace

import pytest

from llm_foundations_companion import cli, preflight
from llm_foundations_companion.control import ControlClient
from llm_foundations_companion.errors import ApiError
from llm_foundations_companion.schema import canonical_json
from llm_foundations_companion.service import Service, storage_error
from test_api import Harness, generation_request, terminal_job
from test_service import inputs


def test_actual_owner_pairing_replaces_previous_bootstrap(tmp_path):
    with Service(tmp_path/'root', *inputs()) as service:
        previous = service.initial_bootstrap.secret
        result = ControlClient(service.root).pair()
        assert result['url'].startswith('http://127.0.0.1:8765/')
        with pytest.raises(ApiError): service.auth.exchange_bootstrap(previous)
        service.close(); service.close()


@pytest.mark.parametrize('outcome', ['accepted', 'failed'])
def test_owner_submit_audits_one_outcome_and_only_job_identity(tmp_path, monkeypatch, outcome):
    with Service(tmp_path/'root', *inputs()) as service:
        job_id = str(uuid.uuid4())
        received = []
        def submit(request, key):
            received.append((request, key))
            if outcome == 'failed': raise ApiError('CAPABILITY_UNAVAILABLE')
            return 202, canonical_json({'job_id': job_id}), False
        monkeypatch.setattr(service, 'submit_job', submit)
        key = str(uuid.uuid4())
        request = generation_request()
        client = ControlClient(service.root)
        if outcome == 'failed':
            with pytest.raises(ApiError) as raised:
                client.call('submit_job', {'request': request, 'idempotency_key': key})
            assert raised.value.code == 'CAPABILITY_UNAVAILABLE'
        else:
            assert client.call('submit_job', {'request': request, 'idempotency_key': key}) == {'job_id': job_id}
        assert len(received) == 1 and received[0][1] == key
        with service.db.read() as connection:
            rows = connection.execute('SELECT operation, outcome, entity_ids_json FROM mutation_audit').fetchall()
        assert len(rows) == 1
        assert tuple(rows[0]) == ('control_submit_job', outcome, json.dumps([job_id] if outcome=='accepted' else [], separators=(',',':')))


def test_owner_cancel_and_event_calls_validate_then_dispatch(tmp_path, monkeypatch):
    with Service(tmp_path/'root', *inputs()) as service:
        job_id = str(uuid.uuid4())
        job = terminal_job(job_id)
        cancelled = []
        monkeypatch.setattr(service, 'cancel_job', lambda identifier: cancelled.append(identifier) or job)
        client = ControlClient(service.root)
        assert client.call('cancel_job', {'job_id': job_id}) == job
        assert cancelled == [job_id]
        pages = []
        page = {'items': [], 'next_after_cursor': 7, 'has_more': False}
        monkeypatch.setattr(service, 'list_events', lambda *args: pages.append(args) or page)
        assert client.call('list_events', {'job_id': job_id, 'after_cursor': 7}) == page
        assert pages == [(job_id, 7, 100)]
        with pytest.raises(ApiError): client.call('list_events', {'job_id': job_id, 'after_cursor': True})
        assert len(pages) == 1


@pytest.mark.parametrize('method', ['_require_registry', '_require_scheduler'])
def test_recovery_mode_missing_components_have_bounded_error(method):
    service = Service.__new__(Service)
    service.registry = service.scheduler = None
    with pytest.raises(ApiError) as raised: getattr(service, method)()
    assert raised.value.code == 'STORAGE_UNAVAILABLE'


def test_scheduler_storage_failure_enters_recovery_without_spinning(tmp_path, monkeypatch):
    with Service(tmp_path/'root', *inputs()) as service:
        service._stop.set(); service._thread.join(timeout=5)
        calls = []
        waits = iter([False, True])
        service._stop = SimpleNamespace(wait=lambda seconds: next(waits), set=lambda: None)
        def fail_dispatch():
            calls.append(True)
            raise sqlite3.OperationalError('private storage path')
        monkeypatch.setattr(service.scheduler, 'dispatch_once', fail_dispatch)
        service._schedule()
        assert calls == [True] and not service.storage_writable


def test_storage_probe_write_failure_is_recorded_without_raw_path(tmp_path, monkeypatch):
    with Service(tmp_path/'root', *inputs()) as service:
        def fail_create(*args): raise OSError('private storage path')
        monkeypatch.setattr(service.registry, 'create_staging_dir', fail_create)
        probe = service._storage_probe()
        assert not probe.writable and not service.storage_writable
        assert 'private' not in repr(probe)
    error = storage_error(SimpleNamespace(code='UNRECOGNIZED_INTERNAL_STORAGE_CODE'))
    assert error.code == 'STORAGE_UNAVAILABLE' and 'UNRECOGNIZED' not in error.message


def test_http_preflight_and_missing_static_or_registry_records_are_bounded():
    h = Harness()
    h.preflight_receipt = {}  # An invalid internal receipt must not escape as a successful response.
    assert h.request('/api/v1/preflight')[0] == 500
    for kind in ['runs', 'models', 'checkpoints', 'tokenizers']:
        status, result, _ = h.request('/api/v1/'+kind+'/'+str(uuid.uuid4()))
        assert status == 404 and result['error']['code'] == 'NOT_FOUND'
    assert h.request('/course/not-a-file.css')[0] == 404


def test_os_failure_before_response_is_redacted_to_internal_error():
    h = Harness()
    def fail_read(*args): raise OSError('private storage path')
    h.list_datasets = fail_read
    status, result, _ = h.request('/api/v1/datasets')
    assert status == 500 and result['error']['code'] == 'INTERNAL_ERROR'
    assert 'private' not in json.dumps(result)


def test_cli_preflight_failure_is_bounded_without_traceback(monkeypatch, capsys):
    def failed(*args, **kwargs):
        raise preflight.PreflightError('BACKEND_VERSION_MISMATCH', 'Backend check failed.')
    monkeypatch.setattr(cli, 'serve', failed)
    assert cli.main(['serve']) == 1
    result = capsys.readouterr()
    assert result.err.strip() == 'BACKEND_VERSION_MISMATCH: Backend check failed.'
    assert not result.out
