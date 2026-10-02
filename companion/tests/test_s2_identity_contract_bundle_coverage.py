"""Reject damaged packaged authority and malformed public error identities."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from llm_foundations_companion import contract_data, errors, runtime_identity
from test_runtime_identity import installed_fixture


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    contract_data._manifest_rows.cache_clear()
    contract_data._load_json_cached.cache_clear()
    monkeypatch.setattr(contract_data.resources, 'files', lambda _: tmp_path)
    yield tmp_path
    contract_data._manifest_rows.cache_clear()
    contract_data._load_json_cached.cache_clear()


def manifest_for(raw=b'{"value":1}'):
    return {'format': 'llm-foundations-runtime-contracts-v1', 'files': [
        {'name': 'test.json', 'source': 'sealed/test.json', 'size_bytes': len(raw),
         'sha256': hashlib.sha256(raw).hexdigest()}]}


def put_bundle(bundle, manifest=None, raw=b'{"value":1}'):
    (bundle/'test.json').write_bytes(raw)
    (bundle/'manifest.json').write_text(json.dumps(manifest or manifest_for(raw)))


@pytest.mark.parametrize('name', ['', '/absolute.json', '../secret.json', 'nested/../../secret.json'])
def test_contract_resource_names_cannot_escape_bundle(bundle, name):
    with pytest.raises(contract_data.ContractDataError, match='invalid bundled'):
        contract_data._resource_bytes(name)


@pytest.mark.parametrize('mode', ['missing', 'directory'])
def test_missing_or_directory_contract_is_a_bounded_bundle_error(bundle, mode):
    if mode == 'directory':
        (bundle/'test.json').mkdir()
    with pytest.raises(contract_data.ContractDataError, match='unavailable'):
        contract_data._resource_bytes('test.json')


@pytest.mark.parametrize('raw', [b'\xff', b'{', b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_sealed_contract_json_rejects_ambiguous_or_nonfinite_values(bundle, raw):
    put_bundle(bundle, raw=raw)
    with pytest.raises(contract_data.ContractDataError, match='invalid JSON'):
        contract_data.load_json('test.json')


@pytest.mark.parametrize('mode', ['list', 'format', 'no_files', 'bad_row', 'name', 'source', 'bool_size', 'negative_size', 'digest', 'duplicate'])
def test_malformed_bundle_manifest_is_rejected_before_contract_load(bundle, mode):
    manifest = manifest_for()
    row = manifest['files'][0]
    if mode == 'list': manifest = []
    elif mode == 'format': manifest['format'] = 'future-v9'
    elif mode == 'no_files': manifest['files'] = []
    elif mode == 'bad_row': manifest['files'] = [None]
    elif mode == 'name': row['name'] = 1
    elif mode == 'source': row['source'] = None
    elif mode == 'bool_size': row['size_bytes'] = True
    elif mode == 'negative_size': row['size_bytes'] = -1
    elif mode == 'digest': row['sha256'] = 'short'
    elif mode == 'duplicate': manifest['files'].append(dict(row))
    (bundle/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(contract_data.ContractDataError):
        contract_data.load_json('test.json')


@pytest.mark.parametrize('mode', ['undeclared', 'size', 'digest'])
def test_bundle_inventory_and_byte_binding_are_enforced(bundle, mode):
    put_bundle(bundle)
    name = 'not-declared.json' if mode == 'undeclared' else 'test.json'
    if mode == 'size': (bundle/name).write_bytes(b'{}')
    if mode == 'digest': (bundle/name).write_bytes(b'{"value":2}')
    with pytest.raises(contract_data.ContractDataError, match='not declared|size mismatch|digest mismatch'):
        contract_data.load_json(name)


def test_bundle_cached_documents_are_defensive_copies(bundle):
    put_bundle(bundle)
    first = contract_data.load_json('test.json'); first['value'] = 200
    assert contract_data.load_json('test.json') == {'value': 1}
    assert contract_data.declared_files() == ('test.json',)


@pytest.mark.parametrize('document', [None, {}, {'codes': [None]}, {'codes': [{'code': 42}]}, {'codes': [{'code': 'X'}, {'code': 'X'}]}])
def test_bad_error_vocabulary_is_rejected_at_load(monkeypatch, document):
    monkeypatch.setattr(errors, 'load_json', lambda _: document)
    with pytest.raises(RuntimeError, match='vocabulary|duplicate'):
        errors._vocabulary()


@pytest.mark.parametrize('field_errors', [[None], [{'field_path': 'x'}], [{'field_path': 1, 'message': 'bad'}],
    [{'field_path': 'x'*501, 'message': 'bad'}], [{'field_path': 'x', 'message': ''}],
    [{'field_path': 'x', 'message': 'm'*1001}], [{'field_path': 'x', 'message': 'bad'}]*101])
def test_public_field_error_bounds_reject_invalid_payloads(field_errors):
    with pytest.raises(ValueError):
        errors.ApiError('VALIDATION_FAILED', field_errors=field_errors)


@pytest.mark.parametrize('mode', ['kind', 'missing_binding', 'nonobject_bindings', 'http_bool', 'retry_string', 'reason_status', 'reason_retry', 'reason_unknown', 'message'])
def test_public_error_binding_cannot_be_reinterpreted(monkeypatch, mode):
    rows = {row['code']: deepcopy(row) for row in errors.load_json('error-vocabulary.json')['codes']}
    monkeypatch.setattr(errors, 'ERROR_VOCABULARY', rows)
    top = rows['VALIDATION_FAILED']
    reason = rows['SCHEMA_INVALID']
    top_binding = top.get('bindings', {}).get('top_level', top)
    reason_binding = reason.get('bindings', {}).get('reason', reason)
    kwargs = {'reason_code': 'SCHEMA_INVALID'}
    exception = RuntimeError
    if mode == 'kind': top['kinds'] = []; exception = ValueError
    elif mode == 'missing_binding': top['bindings'] = {}
    elif mode == 'nonobject_bindings': top['bindings'] = []
    elif mode == 'http_bool': top_binding['http_status'] = True
    elif mode == 'retry_string': top_binding['retryable'] = 'false'
    elif mode == 'reason_status': reason_binding['http_status'] = 500
    elif mode == 'reason_retry': reason_binding['retryable'] = None
    elif mode == 'reason_unknown': kwargs['reason_code'] = 'NONEXISTENT'; exception = ValueError
    elif mode == 'message': kwargs['message'] = 'x'*2001; exception = ValueError
    with pytest.raises(exception):
        errors.ApiError('VALIDATION_FAILED', **kwargs)


@pytest.mark.parametrize('request_id', [None, 'bad', 'AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA'])
def test_public_error_request_id_is_canonical(request_id):
    with pytest.raises(ValueError, match='request_id'):
        errors.ApiError('NOT_FOUND').as_error(request_id)


@pytest.mark.parametrize('mode', ['format', 'missing_profile', 'digest_type', 'digest_hex'])
def test_packaged_profile_lock_rejects_malformed_identity(monkeypatch, mode):
    value = {'format': 'llmf-runtime-lock-manifest-v1', 'profiles': {
        name: {'dependency_lock_sha256': 'a'*64}
        for name in ['win-cpu', 'win-cuda', 'wsl-cpu', 'wsl-cuda']}}
    if mode == 'format': value['format'] = 'unknown'
    elif mode == 'missing_profile': del value['profiles']['win-cpu']
    elif mode == 'digest_type': value['profiles']['wsl-cpu']['dependency_lock_sha256'] = 1
    elif mode == 'digest_hex': value['profiles']['wsl-cpu']['dependency_lock_sha256'] = 'G'*64
    monkeypatch.setattr(runtime_identity, '_read', lambda _: value)
    with pytest.raises(ValueError): runtime_identity.profile_lock('wsl-cpu')


@pytest.mark.parametrize('mode', ['format', 'extra', 'revision', 'tree', 'dirty', 'digest'])
def test_malformed_build_provenance_never_verifies_package(monkeypatch, mode):
    value = {'format': 'llmf-build-provenance-v1', 'source_revision': 'a'*40,
        'source_tree': 'b'*40, 'dirty': False, 'package_source_sha256': 'c'*64}
    if mode == 'format': value['format'] = 'bad'
    elif mode == 'extra': value['extra'] = None
    elif mode == 'revision': value['source_revision'] = 'short'
    elif mode == 'tree': value['source_tree'] = None
    elif mode == 'dirty': value['dirty'] = 0
    elif mode == 'digest': value['package_source_sha256'] = 'Z'*64
    monkeypatch.setattr(runtime_identity, '_read', lambda _: value)
    monkeypatch.setattr(runtime_identity, '_verify_installed_source', lambda _: pytest.fail('malformed identity must fail first'))
    with pytest.raises(ValueError): runtime_identity.build_provenance()


def test_installed_package_source_digest_is_independent_of_wheel_record(tmp_path, monkeypatch):
    root, value, dist = installed_fixture(tmp_path, monkeypatch)
    altered = dict(value, package_source_sha256='0'*64)
    monkeypatch.setattr(runtime_identity, '_read', lambda _: altered)
    with pytest.raises(ValueError, match='build identity'):
        runtime_identity.build_provenance()


def test_bytecode_cache_is_not_part_of_installed_source_identity(tmp_path, monkeypatch):
    root, value, dist = installed_fixture(tmp_path, monkeypatch)
    (root/'__pycache__').mkdir(); (root/'__pycache__/cache.pyc').write_bytes(b'cache')
    assert runtime_identity.build_provenance() == value
