"""Owner command behavior without opening a live service or shared port."""
import json
import uuid
from types import SimpleNamespace

import pytest

from llm_foundations_companion import cli, control, preflight
from llm_foundations_companion.errors import ApiError
from test_api import generation_request
from test_service import inputs


@pytest.fixture
def owner_client(monkeypatch):
    calls = []
    class Client:
        def __init__(self, root):
            calls.append(('root', str(root)))
        def verify_instance(self):
            calls.append(('verify', None))
        def pair(self):
            calls.append(('pair', None))
            return {'url': 'http://127.0.0.1:8765/#test-token'}
        def call(self, operation, arguments):
            calls.append((operation, arguments))
            return {'operation': operation, 'received': arguments}
    monkeypatch.setattr(control, 'ControlClient', Client)
    # The CLI output function is captured at its explicit output boundary.
    outputs = []
    monkeypatch.setattr(cli, 'print_json', lambda value, **kwargs: outputs.append(value))
    return calls, outputs


@pytest.mark.parametrize(('command', 'operation', 'arguments'), [
    (['status'], 'status', {}),
    (['cancel', '--job-id', '11111111-1111-4111-8111-111111111111'],
        'cancel_job', {'job_id': '11111111-1111-4111-8111-111111111111'}),
    (['events', '--job-id', '11111111-1111-4111-8111-111111111111', '--after', '42'],
        'list_events', {'job_id': '11111111-1111-4111-8111-111111111111', 'after_cursor': 42}),
])
def test_owner_read_and_mutation_commands_verify_instance_before_dispatch(owner_client, command, operation, arguments):
    calls, outputs = owner_client
    assert cli.main(command) == 0
    assert calls[1:] == [('verify', None), (operation, arguments)]
    assert outputs == [{'operation': operation, 'received': arguments}]


def test_connect_uses_pairing_without_exposing_an_owner_request(owner_client, capsys):
    calls, outputs = owner_client
    assert cli.main(['connect']) == 0
    assert calls[1:] == [('pair', None)] and outputs == []
    assert capsys.readouterr().out.strip() == 'http://127.0.0.1:8765/#test-token'


@pytest.mark.parametrize('cursor', ['-1', '+1', '1.0', '١', '10000000000'])
def test_events_rejects_noncanonical_or_overlong_cursor(owner_client, cursor):
    calls, outputs = owner_client
    assert cli.main(['events', '--job-id', str(uuid.uuid4()), '--after', cursor]) == 1
    assert calls[1:] == [('verify', None)]
    assert outputs[-1]['error']['reason_code'] == 'SCHEMA_INVALID'


def test_request_preserves_explicit_key_and_normalizes_numbers(owner_client, tmp_path):
    calls, outputs = owner_client
    request = generation_request()
    path = tmp_path/'request.json'; path.write_text(json.dumps(request))
    key = str(uuid.uuid4())
    assert cli.main(['request', '--file', str(path), '--idempotency-key', key]) == 0
    operation, arguments = calls[-1]
    assert operation == 'submit_job' and arguments['idempotency_key'] == key
    assert type(arguments['request']['temperature']) is float
    assert arguments['request']['seed'] == 0


@pytest.mark.parametrize('mode', ['oversize', 'missing', 'duplicate', 'invalid_key'])
def test_request_file_errors_never_dispatch(owner_client, tmp_path, mode):
    calls, outputs = owner_client
    path = tmp_path/'request.json'
    args = ['request', '--file', str(path)]
    if mode == 'oversize':
        path.write_bytes(b' ' * 1048577)
    elif mode == 'duplicate':
        path.write_bytes(b'{"operation":1,"operation":2}')
    elif mode == 'invalid_key':
        path.write_text(json.dumps(generation_request()))
        args += ['--idempotency-key', 'not-a-uuid']
    assert cli.main(args) == 1
    assert calls[1:] == [('verify', None)]
    assert outputs[-1]['error']['code'] == ('STORAGE_UNAVAILABLE' if mode == 'missing' else
        'PAYLOAD_TOO_LARGE' if mode == 'oversize' else 'VALIDATION_FAILED')


def test_gpu_check_emits_offer_and_explanation(monkeypatch, capsys):
    offer = {'status': 'not_detected'}
    outputs = []
    monkeypatch.setattr(preflight, 'gpu_check', lambda: (offer, 'No supported GPU was detected.'))
    monkeypatch.setattr(cli, 'print_json', lambda value: outputs.append(value))
    assert cli.main(['gpu-check']) == 0
    assert outputs == [offer]
    assert capsys.readouterr().out.strip() == 'No supported GPU was detected.'


def test_fixed_port_conflict_closes_listener_without_creating_service(monkeypatch):
    selection, child, query = inputs()
    monkeypatch.setattr(preflight, 'select_profile', lambda profile: selection)
    monkeypatch.setattr(preflight, 'run_gpu_query', lambda kind: query)
    monkeypatch.setattr(preflight, 'run_preflight_child', lambda *args, **kwargs: child)
    calls = []
    class Listener:
        def setsockopt(self, *args):
            calls.append('option')
        def bind(self, endpoint):
            calls.append(endpoint)
            raise OSError('port held')
        def close(self):
            calls.append('closed')
    monkeypatch.setattr(cli.socket, 'socket', lambda *args: Listener())
    with pytest.raises(ApiError) as raised:
        cli.serve(SimpleNamespace(profile='wsl-cpu', storage='unused'))
    assert raised.value.code == 'STORAGE_IN_USE'
    assert calls[-2:] == [('127.0.0.1', 8765), 'closed']
