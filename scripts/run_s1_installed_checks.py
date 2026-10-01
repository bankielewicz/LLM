"""Exercise the installed S1 wheel through real loopback HTTP and owner-only IPC.

This is service integration evidence. It does not execute model operations or
qualify any browser, Windows profile, CUDA profile, or frozen product case.
"""
from __future__ import annotations

import argparse
import base64
import csv
import email.parser
import hashlib
import http.client
import io
import importlib.util
import marshal
import json
import os
from pathlib import Path, PurePosixPath
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile

ORIGIN = 'http://127.0.0.1:8765'
CSP = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'; worker-src 'self'"
PLAN = ('installed-wheel-identity', 'isolated-parent-import',
        'installed-worker-closed-dispatch', 'native-startup-preflight',
        'loopback-health-and-reader', 'ordered-http-security', 'private-ipc-pairing',
        'dataset-http-registration', 'multipart-idempotency', 'artifact-download-boundary',
        'rejected-upload-atomicity', 'inert-bundle-upload', 'unavailable-jobs-create-no-row',
        'session-revocation', 'root-lease-conflict', 'restart-recovery-and-session-expiry')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


WORKER_PROBE = r'''\
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

from llm_foundations_companion.operations import HANDLERS, OPERATIONS, OperationUnavailable
from llm_foundations_companion.scheduler import WorkerController
from llm_foundations_companion.schema import canonical_json

assert tuple(HANDLERS) == OPERATIONS and len(OPERATIONS) == 15
unavailable = 0
for operation, handler in HANDLERS.items():
    try:
        handler({'operation': operation}, object())
    except OperationUnavailable as exc:
        assert exc.code == 'CAPABILITY_UNAVAILABLE'
        unavailable += 1
    else:
        raise AssertionError('S1 installed an executable operation handler: ' + operation)
assert unavailable == len(OPERATIONS)
assert len(set(HANDLERS.values())) == 1

controller = WorkerController()
assert controller.command_prefix == (
    sys.executable, '-I', '-m', 'llm_foundations_companion.worker_main'
)
job_id, instance_id = str(uuid.uuid4()), str(uuid.uuid4())
request = {
    'operation': 'model_prepare',
    'model_profile_id': 'smollm2-135m-instruct-v1',
    'accept_download': True,
}
request_sha256 = hashlib.sha256(canonical_json(request)).hexdigest()
owned = controller.launch(
    job_id=job_id,
    instance_id=instance_id,
    runtime_profile='wsl-cpu',
    root=Path(sys.argv[1]),
    request=request,
    request_sha256=request_sha256,
    schema_id='ModelPrepareRequest',
)
messages = []
try:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        messages.extend(owned.messages())
        if owned.poll() is not None and any(message is None for message in messages):
            break
        time.sleep(0.01)
    else:
        raise AssertionError('Installed worker did not reach protocol EOF within 15 seconds')
    owned.process.wait(timeout=5)
    messages.extend(owned.messages())
    assert len(messages) == 3, messages
    ready, error, eof = messages
    assert ready['type'] == 'ready'
    assert ready['job_id'] == job_id and ready['instance_id'] == instance_id
    assert ready['request_sha256'] == request_sha256
    assert error['type'] == 'error'
    assert error['error']['code'] == 'WORKER_PROTOCOL_ERROR'
    assert eof is None
    assert owned.ready and owned.terminal and owned.protocol_eof
    assert owned.process.returncode == 2
finally:
    if owned.poll() is None:
        owned.terminate_owned(owned.spawn_nonce)
        owned.process.wait(timeout=5)
    owned.close()
assert owned.stdout is not None and owned.stdout.total_bytes == 0
assert owned.stderr is not None and owned.stderr.total_bytes == 0
print(json.dumps({
    'command_prefix': list(controller.command_prefix),
    'operation_count': len(OPERATIONS),
    'unavailable_handlers': unavailable,
    'protocol_messages': ['ready', 'error', 'eof'],
    'error_code': error['error']['code'],
    'exit_code': owned.process.returncode,
}, sort_keys=True))
'''


class Installed:
    def __init__(self, python, wheel, report):
        self.python = python.absolute()
        self.wheel = wheel.resolve()
        self.executable_sha256 = sha(self.python.read_bytes())
        self.wheel_sha256 = sha(self.wheel.read_bytes())
        self.environment = {k: v for k, v in os.environ.items() if k in {'PATH', 'HOME', 'USER', 'LANG', 'LC_ALL', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP'}}
        self.environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
        self.report = report
        self.rows = []
        self.purelib = None
        self.package = None
        self.process = None
        self.stderr = []
        self.stdout = queue.Queue()
        self.session = None
        self.root = None

    def identity(self):
        assert sha(self.python.read_bytes()) == self.executable_sha256, 'Runtime interpreter changed'
        assert sha(self.wheel.read_bytes()) == self.wheel_sha256, 'Candidate wheel changed'
        layout = subprocess.run(
            [str(self.python), '-I', '-c', "import json,sysconfig; print(json.dumps({'purelib':sysconfig.get_path('purelib')}))"],
            cwd=self.wheel.parent, env=self.environment, capture_output=True, timeout=15)
        assert layout.returncode == 0 and not layout.stderr, 'Could not locate installed runtime files'
        purelib = Path(json.loads(layout.stdout)['purelib']).resolve(strict=True)
        assert purelib.is_dir(), 'Installed runtime package directory is unavailable'
        if self.purelib is None:
            self.purelib = purelib
            self.package = purelib / 'llm_foundations_companion'
        assert purelib == self.purelib, 'Installed runtime package directory changed'
        assert self.package is not None and self.package.is_dir() and not self.package.is_symlink(), 'Installed package root is unsafe'
        count = 0
        expected_package = set()
        with zipfile.ZipFile(self.wheel) as archive:
            names = archive.namelist()
            assert len(names) == len(set(names)), 'Candidate wheel has duplicate members'
            assert all('\\' not in name and not name.startswith('/') and '..' not in PurePosixPath(name).parts for name in names), 'Candidate wheel has an unsafe member path'
            metadata_members = [name for name in names if name.endswith('.dist-info/METADATA')]
            assert len(metadata_members) == 1, 'Candidate wheel has ambiguous distribution metadata'
            dist_prefix = metadata_members[0].split('/', 1)[0]
            dist_prefixes = {name.split('/', 1)[0] for name in names if '.dist-info/' in name}
            assert dist_prefixes == {dist_prefix}, 'Candidate wheel has ambiguous dist-info members'
            metadata = email.parser.BytesParser().parsebytes(archive.read(metadata_members[0]))
            assert metadata['Name'] == 'llm-foundations-companion' and metadata['Version'] == '0.1.0', 'Candidate wheel metadata identity differs'
            expected_dist = {name for name in names if name.startswith(dist_prefix + '/') and not name.endswith('/')}
            assert dist_prefix + '/RECORD' in expected_dist, 'Candidate wheel has no RECORD'
            for member in names:
                if not member.startswith('llm_foundations_companion/') or member.endswith('/'):
                    continue
                expected_package.add(member)
                target = purelib / member
                assert target.is_file() and target.read_bytes() == archive.read(member), 'Installed package differs: ' + member
                count += 1
            for member in expected_dist - {dist_prefix + '/RECORD'}:
                target = purelib / member
                assert target.is_file() and target.read_bytes() == archive.read(member), 'Installed wheel metadata differs: ' + member
        dist_path = purelib / dist_prefix
        assert dist_path.is_dir() and not dist_path.is_symlink(), 'Installed dist-info root is unsafe'
        other_metadata = set(purelib.glob('llm_foundations_companion-*.dist-info')) | set(purelib.glob('llm_foundations_companion-*.egg-info'))
        assert other_metadata == {dist_path}, 'Multiple installed candidate metadata directories exist'
        observed = set()
        for target in self.package.rglob('*'):
            assert not target.is_symlink(), 'Installed package contains a symlink: ' + str(target)
            if target.is_dir():
                continue
            assert target.is_file(), 'Installed package contains a non-regular entry: ' + str(target)
            installed_name = target.relative_to(purelib).as_posix()
            observed.add(installed_name)
            if target.suffix == '.pyc' and target.parent.name == '__pycache__':
                source = target.parent.parent / (target.name.split('.')[0] + '.py')
                assert source.is_file(), 'Orphan installed bytecode cache'
                raw = target.read_bytes()
                assert raw[:4] == importlib.util.MAGIC_NUMBER, 'Bytecode interpreter mismatch'
                assert marshal.loads(raw[16:]) == compile(source.read_bytes(), str(source), 'exec', dont_inherit=True), 'Installed bytecode differs from source'
            else:
                assert installed_name in expected_package, 'Unexpected installed companion file: ' + installed_name
        pip_metadata = {dist_prefix + '/' + name for name in ('INSTALLER', 'REQUESTED', 'direct_url.json')}
        allowed_dist = expected_dist | pip_metadata
        for target in dist_path.rglob('*'):
            assert not target.is_symlink(), 'Installed dist-info contains a symlink: ' + str(target)
            if target.is_dir():
                continue
            assert target.is_file(), 'Installed dist-info contains a non-regular entry: ' + str(target)
            installed_name = target.relative_to(purelib).as_posix()
            assert installed_name in allowed_dist, 'Unexpected installed dist-info file: ' + installed_name
            observed.add(installed_name)
        assert pip_metadata <= observed and expected_dist <= observed, 'Installed distribution metadata is incomplete'
        record_path = dist_path / 'RECORD'
        rows = list(csv.reader(io.StringIO(record_path.read_text(encoding='utf-8'), newline='')))
        assert all(len(row) == 3 and row[0] for row in rows), 'Installed RECORD has an invalid row'
        assert len(rows) == len({row[0] for row in rows}), 'Installed RECORD has duplicate paths'
        assert {row[0] for row in rows} == observed, 'Installed RECORD does not describe the exact installed distribution'
        for name, digest, size in rows:
            relative = PurePosixPath(name)
            assert not relative.is_absolute() and '..' not in relative.parts, 'Installed RECORD path escapes purelib'
            target = purelib.joinpath(*relative.parts)
            if digest:
                algorithm, encoded = digest.split('=', 1)
                assert algorithm == 'sha256'
                actual = base64.urlsafe_b64encode(hashlib.sha256(target.read_bytes()).digest()).rstrip(b'=').decode('ascii')
                assert actual == encoded, 'Installed RECORD digest differs: ' + name
            if size:
                assert target.stat().st_size == int(size), 'Installed RECORD size differs: ' + name
        assert count > 30, 'Candidate wheel is incomplete'
        return {'wheel_sha256': self.wheel_sha256, 'python_sha256': self.executable_sha256,
            'purelib': str(purelib), 'verified_package_members': count,
            'verified_distribution_files': len(observed)}

    def worker_probe(self):
        self.identity()
        probe_root = self.root.parent / 'installed-worker-probe'
        result = subprocess.run([str(self.python), '-I', '-c', WORKER_PROBE, str(probe_root)],
            cwd=self.root.parent, env=self.environment, capture_output=True, timeout=30)
        assert result.returncode == 0, (result.stderr or result.stdout).decode('utf-8', 'replace')[:1500]
        assert not result.stderr, result.stderr.decode('utf-8', 'replace')[:1500]
        return json.loads(result.stdout)

    def command(self, *arguments):
        self.identity()
        return subprocess.run([str(self.python), '-I', '-m', 'llm_foundations_companion', *arguments],
            cwd=self.root.parent, env=self.environment, capture_output=True, timeout=45)

    def check(self, name, action):
        start = time.time()
        try:
            observation = action()
            self.rows.append({'check': name, 'status': 'PASS', 'seconds': round(time.time() - start, 3), 'observation': observation})
        except Exception as exc:
            self.rows.append({'check': name, 'status': 'FAIL', 'seconds': round(time.time() - start, 3), 'error': type(exc).__name__ + ': ' + str(exc)[:1500]})
            raise

    def start(self):
        self.identity()
        self.stdout = queue.Queue()
        self.stderr = []
        self.process = subprocess.Popen([str(self.python), '-I', '-m', 'llm_foundations_companion', 'serve', '--storage', str(self.root), '--profile', 'wsl-cpu'],
            cwd=self.root.parent, env=self.environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        def stdout_reader():
            for line in self.process.stdout:
                self.stdout.put(line.decode('utf-8', 'replace').strip())
        def stderr_reader():
            for block in iter(lambda: self.process.stderr.read(4096), b''):
                if sum(map(len, self.stderr)) < 65536:
                    self.stderr.append(block)
        self.readers = [threading.Thread(target=stdout_reader, daemon=True), threading.Thread(target=stderr_reader, daemon=True)]
        for thread in self.readers:
            thread.start()
        deadline = time.monotonic() + 45
        bootstrap = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise AssertionError('Service exited during startup: ' + b''.join(self.stderr).decode('utf-8', 'replace')[:1500])
            try:
                line = self.stdout.get(timeout=0.1)
            except queue.Empty:
                continue
            if line.startswith(ORIGIN + '/#/connect/'):
                bootstrap = line.rsplit('/', 1)[1]
                break
        assert bootstrap is not None, 'Service did not publish a pairing grant'
        self.session = self.exchange(bootstrap)
        return self.session

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
                raise AssertionError('Service did not shut down within the cancellation bound')
            for thread in self.readers:
                thread.join(timeout=2)
            self.process = None

    def http(self, method, path, *, body=None, headers=None, auth=True):
        values = {'Host': '127.0.0.1:8765', 'Sec-Fetch-Site': 'none'}
        if auth and self.session:
            values['Authorization'] = 'Bearer ' + self.session['access_token']
        if method in {'POST', 'PUT', 'PATCH', 'DELETE'}:
            values['Origin'] = ORIGIN
            if auth and self.session:
                values['X-LLMF-CSRF'] = self.session['csrf_token']
            values['Idempotency-Key'] = str(uuid.uuid4())
        if body is not None and not isinstance(body, bytes):
            body = canonical(body)
            values['Content-Type'] = 'application/json'
        values.update(headers or {})
        values = {key: value for key, value in values.items() if value is not None}
        connection = http.client.HTTPConnection('127.0.0.1', 8765, timeout=10)
        try:
            connection.request(method, path, body, values)
            response = connection.getresponse()
            raw, response_headers = response.read(), dict(response.getheaders())
            return response.status, raw, {k.lower(): v for k, v in response_headers.items()}
        finally:
            connection.close()

    def exchange(self, secret):
        status, raw, _ = self.http('POST', '/api/v1/sessions', body={'bootstrap_secret': secret}, auth=False)
        assert status == 201, raw
        return json.loads(raw)

    def get(self, path):
        status, raw, _ = self.http('GET', path)
        assert status == 200, raw
        return json.loads(raw)

    @staticmethod
    def multipart(parts):
        boundary = 'llmf-' + uuid.uuid4().hex
        output = bytearray()
        for name, media, raw in parts:
            output.extend(('--' + boundary + '\r\nContent-Disposition: form-data; name="' + name + '"; filename="ignored.bin"\r\nContent-Type: ' + media + '\r\n\r\n').encode())
            output.extend(raw)
            output.extend(b'\r\n')
        output.extend(('--' + boundary + '--\r\n').encode())
        return bytes(output), {'Content-Type': 'multipart/form-data; boundary=' + boundary}


def run(t):
    t.check('installed-wheel-identity', t.identity)
    def parent_import():
        t.identity()
        code = "import json,sys; import llm_foundations_companion.service, llm_foundations_companion.api; assert not ({'torch','transformers','peft'} & sys.modules.keys()); print(json.dumps({'ml_modules_imported':False}))"
        result = subprocess.run([str(t.python), '-I', '-c', code], cwd=t.root.parent, env=t.environment, capture_output=True, timeout=15)
        assert result.returncode == 0, result.stderr.decode()[:1500]
        return json.loads(result.stdout)
    t.check('isolated-parent-import', parent_import)
    t.check('installed-worker-closed-dispatch', t.worker_probe)
    def startup():
        t.start()
        receipt = t.get('/api/v1/preflight')
        assert receipt['status'] == 'pass', receipt
        assert receipt['receipt_artifact_id'] is not None
        stored = t.http('GET', '/api/v1/artifacts/' + receipt['receipt_artifact_id'] + '/content')
        assert stored[0] == 200 and json.loads(stored[1]) == receipt
        observed = dict(receipt)
        observed.pop('receipt_artifact_id'); digest = observed.pop('receipt_sha256')
        assert sha(canonical(observed)) == digest
        return {'profile': receipt['profile'], 'tests': receipt['tests'], 'receipt_sha256': digest}
    t.check('native-startup-preflight', startup)
    def reader():
        status, raw, headers = t.http('GET', '/healthz', auth=False)
        assert status == 200 and json.loads(raw) == {'status': 'ok', 'api_version': '1.0'}
        status, raw, headers = t.http('GET', '/', auth=False)
        assert status == 200 and headers['content-security-policy'] == CSP
        assert headers['x-content-type-options'] == 'nosniff'
        return {'health': 'minimal', 'reader_csp': 'exact'}
    t.check('loopback-health-and-reader', reader)
    def security():
        cases = [('OPTIONS', {'Host': 'localhost:8765'}, 403, 'HOST_MISMATCH'), ('OPTIONS', {}, 403, 'PREFLIGHT_REJECTED'), ('POST', {'Origin': 'https://example.invalid'}, 403, 'ORIGIN_MISMATCH'), ('GET', {'Sec-Fetch-Site': 'cross-site'}, 403, 'FETCH_SITE_REJECTED'), ('GET', {'Authorization': None}, 401, None), ('POST', {'X-LLMF-CSRF': 'bad'}, 403, None)]
        for method, headers, expected, reason in cases:
            status, raw, response_headers = t.http(method, '/api/v1/jobs', body=b'not-json' if method=='POST' else None, headers=headers)
            assert status == expected, raw
            if reason:
                assert json.loads(raw)['error']['reason_code'] == reason
            assert not any(name.startswith('access-control') for name in response_headers)
        return {'ordered_cases': len(cases)}
    t.check('ordered-http-security', security)
    def ipc():
        result = t.command('status', '--storage', str(t.root))
        assert result.returncode == 0, result.stderr
        status = json.loads(result.stdout)
        assert status['pid'] == t.process.pid and status['instance_id'] == t.session['instance_id']
        result = t.command('connect', '--storage', str(t.root))
        assert result.returncode == 0, result.stderr
        url = result.stdout.decode().strip()
        assert url.startswith(ORIGIN + '/#/connect/')
        t.session = t.exchange(url.rsplit('/', 1)[1])
        if os.name != 'nt':
            assert (t.root/'runtime/control.json').stat().st_mode & 0o777 == 0o600
            assert (t.root/'runtime/control.sock').stat().st_mode & 0o777 == 0o600
        return {'status_matches_owned_pid': True, 'pair_exchange': 'PASS'}
    t.check('private-ipc-pairing', ipc)
    train = canonical({'record_id': 'train-one', 'scenario_group_id': 'group-train', 'text': 'A useful training record with unique content.'}) + b'\n'
    validation = canonical({'record_id': 'validation-one', 'scenario_group_id': 'group-validation', 'text': 'A distinct validation example for service checks.'}) + b'\n'
    parts = [('metadata', 'application/json', canonical({'name': 'S1 integration dataset', 'record_format': 'document_text_v1'})), ('train', 'application/x-ndjson', train), ('validation', 'application/x-ndjson', validation)]
    body, headers = t.multipart(parts)
    key = str(uuid.uuid4()); headers['Idempotency-Key'] = key
    dataset = {}
    first = {}
    def registration():
        status, raw, _ = t.http('POST', '/api/v1/datasets', body=body, headers=headers)
        assert status == 201, raw
        dataset.update(json.loads(raw)); first['bytes'] = raw
        assert dataset['eligibility'] == 'eligible' and dataset['splits']['train']['records'] == 1
        assert t.get('/api/v1/datasets/' + dataset['dataset_id']) == dataset
        return {'dataset_id': dataset['dataset_id'], 'manifest_sha256': dataset['manifest_sha256']}
    t.check('dataset-http-registration', registration)
    def replay():
        status, raw, response_headers = t.http('POST', '/api/v1/datasets', body=body, headers=headers)
        assert status == 201 and raw == first['bytes'] and response_headers['idempotent-replayed'] == 'true'
        changed_parts = list(parts); changed_parts[0] = ('metadata', 'application/json', canonical({'name': 'Changed name', 'record_format': 'document_text_v1'}))
        changed_body, changed_headers = t.multipart(changed_parts); changed_headers['Idempotency-Key'] = key
        status, raw, _ = t.http('POST', '/api/v1/datasets', body=changed_body, headers=changed_headers)
        assert status == 409 and json.loads(raw)['error']['code'] == 'IDEMPOTENCY_CONFLICT'
        assert len(t.get('/api/v1/datasets')['items']) == 1
        return {'exact_response_replayed': True, 'conflicting_body': 409, 'dataset_count': 1}
    t.check('multipart-idempotency', replay)
    def download():
        artifact_id = dataset['splits']['train']['artifact_id']
        descriptor = t.get('/api/v1/artifacts/' + artifact_id)
        status, raw, headers = t.http('GET', '/api/v1/artifacts/' + artifact_id + '/content')
        assert status == 200 and raw == train and sha(raw) == descriptor['sha256']
        assert headers['x-content-type-options'] == 'nosniff'
        assert t.http('GET', '/api/v1/artifacts/' + artifact_id + '/content', headers={'Range':'bytes=0-9'})[0] == 416
        return {'digest_verified': True, 'range_rejected': True}
    t.check('artifact-download-boundary', download)
    def rejected():
        count = len(t.get('/api/v1/artifacts')['items'])
        bad_parts = list(parts); bad_parts[1] = ('train', 'application/x-ndjson', b'{"unknown":true}\n')
        bad, bad_headers = t.multipart(bad_parts)
        status, raw, _ = t.http('POST', '/api/v1/datasets', body=bad, headers=bad_headers)
        assert status == 400, raw
        assert len(t.get('/api/v1/datasets')['items']) == 1
        assert len(t.get('/api/v1/artifacts')['items']) == count
        return {'no_partial_dataset_or_artifact': True}
    t.check('rejected-upload-atomicity', rejected)
    def bundle():
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_STORED) as archive:
            archive.writestr('llm-foundations-bundle/manifest.json', b'{}\n')
            archive.writestr('llm-foundations-bundle/objects/note/id/readme.txt',
                b'This is inert and is not a semantically valid model bundle.')
        raw_bundle = buffer.getvalue()
        upload, upload_headers = t.multipart([('bundle', 'application/vnd.llm-foundations.bundle+zip', raw_bundle)])
        status, raw, _ = t.http('POST', '/api/v1/bundle-uploads', body=upload, headers=upload_headers)
        assert status == 201, raw
        descriptor = json.loads(raw)
        content = t.http('GET', '/api/v1/artifacts/' + descriptor['artifact_id'] + '/content')
        assert content[0] == 200 and content[1] == raw_bundle and content[2]['content-disposition'].startswith('attachment')
        return {'inert_transport_only': True, 'sha256': descriptor['sha256']}
    t.check('inert-bundle-upload', bundle)
    def unavailable():
        request = {'operation': 'tokenizer_train', 'dataset_id': dataset['dataset_id'], 'vocab_size': 257, 'seed': 7, 'tokenizer_profile_id': 'byte-v1'}
        # Use an authoritative fixture shape supplied by the frozen OpenAPI.
        request_file = t.report / 'unavailable-request.json'
        request_file.write_bytes(canonical(request))
        status, raw, _ = t.http('POST', '/api/v1/jobs', body=request)
        assert status == 503 and json.loads(raw)['error']['code'] == 'CAPABILITY_UNAVAILABLE', raw
        result = t.command('request', '--storage', str(t.root), '--file', str(request_file))
        assert result.returncode == 1 and json.loads(result.stderr)['error']['code'] == 'CAPABILITY_UNAVAILABLE', result.stderr
        assert t.get('/api/v1/jobs')['items'] == []
        return {'http': 503, 'control': 'CAPABILITY_UNAVAILABLE', 'job_count': 0}
    t.check('unavailable-jobs-create-no-row', unavailable)
    def revoke():
        status, raw, _ = t.http('DELETE', '/api/v1/sessions/current')
        assert status == 204 and raw == b''
        assert t.http('GET', '/api/v1/runtime')[0] == 401
        result = t.command('connect', '--storage', str(t.root)); assert result.returncode == 0
        t.session = t.exchange(result.stdout.decode().strip().rsplit('/', 1)[1])
        return {'revoked_session_rejected': True}
    t.check('session-revocation', revoke)
    def occupied():
        result = t.command('serve', '--storage', str(t.root), '--profile', 'wsl-cpu')
        assert result.returncode != 0
        assert t.process.poll() is None and t.get('/api/v1/runtime')['storage_writable']
        return {'second_instance_refused': True, 'original_healthy': True}
    t.check('root-lease-conflict', occupied)
    def restart():
        previous = dict(t.session)
        old_instance = previous['instance_id']
        t.stop(); t.start()
        assert t.session['instance_id'] != old_instance
        status, _, _ = t.http('GET', '/api/v1/runtime', headers={'Authorization': 'Bearer ' + previous['access_token']})
        assert status == 401
        assert t.get('/api/v1/datasets/' + dataset['dataset_id']) == dataset
        assert t.get('/api/v1/jobs')['items'] == []
        return {'new_instance': True, 'old_session_rejected': True, 'dataset_preserved': True}
    t.check('restart-recovery-and-session-expiry', restart)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=False)
    t = Installed(args.python, args.wheel, args.report_dir)
    try:
        with tempfile.TemporaryDirectory(prefix='llmf-s1-') as workspace:
            t.root = Path(workspace) / 'root'
            try:
                run(t)
            finally:
                t.stop()
    except Exception:
        pass  # Exact failed check and the full mandatory denominator are retained below.
    seen = {row['check'] for row in t.rows}
    t.rows.extend({'check': name, 'status': 'NOT_RUN', 'reason': 'An earlier required check failed.'} for name in PLAN if name not in seen)
    passed = len(t.rows) == len(PLAN) and all(row['status'] == 'PASS' for row in t.rows)
    result = {'format': 'llm-foundations-s1-installed-checks-v1', 'status': 'PASS' if passed else 'FAIL',
        'wheel_sha256': t.wheel_sha256, 'python_sha256': t.executable_sha256, 'profile': 'wsl-cpu',
        'checks': t.rows, 'product_qualification': 'NOT_RUN'}
    (args.report_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for row in t.rows:
        print(row['check'] + ': ' + row['status'])
        if row.get('error'):
            print(row['error'])
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
