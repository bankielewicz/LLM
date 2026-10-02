"""Service lifecycle controls; backend reports are injected test evidence."""
import hashlib
import sqlite3
import sys
import uuid

import pytest

from llm_foundations_companion.control import ControlClient
from llm_foundations_companion.database import Database
from llm_foundations_companion.errors import ApiError
from llm_foundations_companion.preflight import ChildExecution, GpuQueryResult, ProcessResult, ProfileSelection, ProfileSelectionError
from llm_foundations_companion.schema import canonical_json, strict_json
from llm_foundations_companion.service import Service


def inputs():
    selection = ProfileSelection('wsl-cpu', 'wsl', 'cpu', False, '2.8.0+cpu', None)
    report = {'format': 'llm-foundations-preflight-child-v1', 'backend_status': 'passed', 'device_available': True, 'versions': {'torch': '2.8.0+cpu', 'transformers': '4.57.1', 'peft': '0.17.1', 'accelerate': '1.10.1', 'safetensors': '0.6.2'}, 'error_text': '', 'cuda': None}
    child = ChildExecution(report, '', ProcessResult(0), True)
    query = GpuQueryResult(None, None, 'CUDA_DEVICE_NOT_FOUND', ProcessResult(1))
    return selection, child, query


def test_startup_receipt_is_self_consistent_and_service_imports_no_models(tmp_path):
    before = {'torch', 'transformers', 'peft'} & sys.modules.keys()
    with Service(tmp_path/'root', *inputs()) as service:
        receipt = service.preflight_receipt
        assert receipt['status'] == 'pass'
        descriptor, stream = service.open_artifact(receipt['receipt_artifact_id'])
        with stream:
            raw = stream.read()
        assert strict_json(raw) == receipt
        assert descriptor['sha256'] == hashlib.sha256(raw).hexdigest()
        without = dict(receipt); digest = without.pop('receipt_sha256'); without.pop('receipt_artifact_id')
        assert hashlib.sha256(canonical_json(without)).hexdigest() == digest
        assert all(not capability['available'] for capability in service.runtime_info()['capabilities'].values())
        assert service.companion_source_revision is None  # Source tests have no installed build identity.
        assert {'torch', 'transformers', 'peft'} & sys.modules.keys() == before


def test_root_lease_conflict_does_not_disturb_existing_owner(tmp_path):
    root = tmp_path/'root'
    with Service(root, *inputs()) as service:
        original = ControlClient(root).verify_instance()
        with pytest.raises(ApiError) as raised:
            Service(root, *inputs())
        assert raised.value.code == 'STORAGE_IN_USE'
        assert ControlClient(root).verify_instance()['instance_id'] == original['instance_id']
        assert service.storage_writable


@pytest.mark.parametrize('corruption', ['future_schema', 'invalid_bytes'])
def test_read_only_recovery_preserves_metadata_and_exposes_failed_receipt(tmp_path, corruption):
    root = tmp_path/'root'; root.mkdir(mode=0o700)
    path = root/'metadata.sqlite3'
    if corruption == 'future_schema':
        db = Database(root); db.initialize(); db.close()
        with sqlite3.connect(path) as conn:
            conn.execute('PRAGMA user_version = 99')
    else:
        path.write_bytes(b'this is not a SQLite database\n')
    original = path.read_bytes()
    with Service(root, *inputs()) as service:
        assert not service.storage_writable
        assert service.preflight_receipt['status'] == 'fail'
        assert service.preflight_receipt['receipt_artifact_id'] is None
        assert not ControlClient(root).verify_instance()['storage_writable']
        with pytest.raises(ApiError) as raised:
            service.begin_upload(100, '/api/v1/datasets')
        assert raised.value.code == 'STORAGE_UNAVAILABLE'
        assert path.read_bytes() == original
    assert path.read_bytes() == original


def test_cuda_failure_precedes_creation_of_root_or_lock(tmp_path):
    _, child, query = inputs()
    selection = ProfileSelection('wsl-cuda', 'wsl', 'cuda', True, '2.8.0+cu128', None)
    root = tmp_path/'not-created'
    with pytest.raises(ProfileSelectionError):
        Service(root, selection, child, query)
    assert not root.exists()


def test_session_revocation_idempotency_is_instance_scoped(tmp_path):
    with Service(tmp_path/'root', *inputs()) as service:
        first = service.auth.exchange_bootstrap(service.initial_bootstrap.secret)
        grant = service.auth.issue_bootstrap()
        second = service.auth.exchange_bootstrap(grant.secret)
        def identity(credentials):
            return service.auth.authorize('GET', {'host': '127.0.0.1:8765', 'sec-fetch-site': 'none', 'authorization': 'Bearer ' + credentials.access_token})
        key = str(uuid.uuid4())
        assert service.revoke_session(identity(first), key) == (204, b'', False)
        assert service.revoke_session(identity(second), key) == (204, b'', True)
        assert identity(second) is not None  # Exact replay creates no second side effect.
        with pytest.raises(ApiError):
            identity(first)


def test_probe_cleanup_failure_keeps_reader_in_read_only_recovery(tmp_path, monkeypatch):
    def fail_cleanup(self, path):
        raise OSError('injected cleanup failure')
    monkeypatch.setattr(Service, '_remove_staging', fail_cleanup)
    with Service(tmp_path/'root', *inputs()) as service:
        assert not service.storage_writable
        assert service.preflight_receipt['status'] == 'fail'
        assert service.preflight_receipt['receipt_artifact_id'] is None
        assert ControlClient(service.root).verify_instance()['mode'] == 'local'


def test_cli_unsupported_platform_is_bounded_without_traceback(monkeypatch, capsys):
    from llm_foundations_companion import cli, preflight
    def unsupported(*args, **kwargs):
        raise preflight.UnsupportedPlatformError('BACKEND_VERSION_MISMATCH', 'This operating system is unsupported.')
    monkeypatch.setattr(preflight, 'select_profile', unsupported)
    assert cli.main(['serve']) == 1
    output = capsys.readouterr()
    assert output.err.strip() == 'BACKEND_VERSION_MISMATCH: This operating system is unsupported.'
    assert not output.out


def test_readable_recovery_keeps_healthy_artifacts_and_lists(tmp_path, monkeypatch):
    root = tmp_path/'root'
    with Service(root, *inputs()) as service:
        artifact_id = service.preflight_receipt['receipt_artifact_id']
        original = service.get_artifact(artifact_id)
    from llm_foundations_companion.registry import Registry
    def readable_recovery(self):
        self.db.enter_read_only_recovery('ARTIFACT_RECOVERY_FAILED')
    monkeypatch.setattr(Registry, 'recover', readable_recovery)
    with Service(root, *inputs()) as service:
        assert not service.storage_writable
        assert service.get_artifact(artifact_id) == original
        descriptor, stream = service.open_artifact(artifact_id)
        with stream:
            assert hashlib.sha256(stream.read()).hexdigest() == descriptor['sha256']
        assert service.list_datasets({'limit': 50})['items'] == []
        assert service.list_jobs({'limit': 50})['items'] == []
        assert service._thread is None


def test_control_schema_normalizes_number_fields():
    normalized = Service._control_args({'value': 1}, {'value': {'type': 'number'}}, ['value'])
    assert type(normalized['value']) is float


def test_failed_worker_shutdown_retains_lease_but_close_can_retry(tmp_path, monkeypatch):
    service = Service(tmp_path/'root', *inputs())
    shutdown = service.scheduler.shutdown
    calls = []
    def fail_once():
        calls.append(True)
        if len(calls) == 1:
            raise OSError('injected ownership cleanup failure')
        shutdown()
    monkeypatch.setattr(service.scheduler, 'shutdown', fail_once)
    with pytest.raises(OSError):
        service.close()
    assert not service._closed
    with pytest.raises(ApiError) as raised:
        Service(tmp_path/'root', *inputs())
    assert raised.value.code == 'STORAGE_IN_USE'
    service.close()
    assert service._closed
    with Service(tmp_path/'root', *inputs()):
        pass


def test_control_stop_failure_still_releases_other_resources(tmp_path, monkeypatch):
    root = tmp_path/'root'
    service = Service(root, *inputs())
    stop = service.control.stop
    def stop_then_fail():
        stop()
        raise OSError('injected post-stop cleanup failure')
    monkeypatch.setattr(service.control, 'stop', stop_then_fail)
    with pytest.raises(OSError):
        service.close()
    assert service._closed
    with Service(root, *inputs()):
        pass


def test_control_submit_preserves_schema_then_reference_then_semantic_order():
    service = Service.__new__(Service)
    admitted = []
    def reject_missing_reference(request, key):
        admitted.append(request)
        raise ApiError('NOT_FOUND', 'The dataset is absent.')
    service.submit_job = reject_missing_reference
    request = {
        'operation': 'tiny_train',
        'dataset_id': '11111111-1111-4111-8111-111111111111',
        'tokenizer_id': '22222222-2222-4222-8222-222222222222',
        'architecture_profile_id': 'tiny-v2-standard-v1',
        'steps': 1, 'eval_every': 2, 'batch_size': 1,
        'learning_rate': 0.001, 'seed': 17,
        'context': 8, 'width': 16, 'heads': 1, 'layers': 1,
    }
    envelope = {'request': request, 'idempotency_key': str(uuid.uuid4())}
    with pytest.raises(ApiError) as reference:
        service._control_submit(envelope)
    assert reference.value.code == 'NOT_FOUND'
    assert len(admitted) == 1
    with pytest.raises(ApiError) as schema:
        service._control_submit({**envelope, 'request': {**request, 'steps': 1.0}})
    assert schema.value.reason_code == 'SCHEMA_INVALID'
    assert len(admitted) == 1


def test_invalid_installed_source_identity_precedes_storage_creation(tmp_path, monkeypatch):
    from llm_foundations_companion import service as service_module
    def invalid():
        raise ValueError('private source detail')
    monkeypatch.setattr(service_module, 'source_revision', invalid)
    root = tmp_path / 'not-created'
    with pytest.raises(ProfileSelectionError) as raised:
        Service(root, *inputs())
    assert raised.value.reason_code == 'BACKEND_VERSION_MISMATCH'
    assert 'private' not in raised.value.message
    assert not root.exists()
