"""One root-owned service instance; model imports stay in owned children."""
from __future__ import annotations

import contextlib
import hashlib
import io
import os
import secrets
import shutil
import sqlite3
import threading
import uuid
from pathlib import Path

from . import __version__
from .auth import AuthManager, generate_instance_id
from .control import ControlServer
from .database import Database, DatabaseError, utc_now
from .errors import ApiError
from .limits import LIMITS
from .platform_security import InstanceLease, StorageSecurityError, resolve_storage_root
from .preflight import (StorageProbeResult, build_preflight_receipt, capabilities_from_preflight,
                        detect_cuda_environment_installed, gpu_offer, require_startup_allowed)
from .schema import canonical_json, strict_json, validate_schema
from .transport import inspect_zip_archive


def storage_error(exc):
    code = getattr(exc, 'code', '')
    if code == 'STORAGE_IN_USE':
        return ApiError('STORAGE_IN_USE')
    return ApiError('STORAGE_UNAVAILABLE', 'The storage root is unavailable or in recovery mode.')


class Service:
    """Serialize startup recovery before publishing sessions or control IPC."""
    def __init__(self, root, selection, child, query):
        require_startup_allowed(selection, child, query)
        self.selection, self.child, self.query = selection, child, query
        self.instance_id = generate_instance_id()
        self.started_at = utc_now()
        self.root = Path(root)
        self.db = self.registry = self.datasets = self.scheduler = self.idempotency = None
        self.control = self.lease = None
        self._thread = None
        self._stop = threading.Event()
        self._upload_lock = threading.RLock()
        self._closed = False
        self.capabilities = capabilities_from_preflight(selection, child)
        self.gpu_offer = gpu_offer(selection, query, cuda_environment_installed=detect_cuda_environment_installed(selection))
        try:
            self.root = resolve_storage_root(self.root, create=True)
            self.lease = InstanceLease(self.root).acquire()
            self.db = Database(self.root)
            self.db.initialize()
            if self.db.installation_id is not None:
                from .registry import Registry
                from .datasets import DatasetRegistry
                from .scheduler import IdempotencyStore, Scheduler
                self.registry = Registry(self.db, self.root, self.instance_id, secrets.token_bytes(32))
                if not self.db.read_only:
                    self.registry.recover()
                self.datasets = DatasetRegistry(self.registry)
                with self.db.read() as connection:
                    scheduler_schema = connection.execute("SELECT version FROM schema_components WHERE component = 'scheduler'").fetchone()
                if not self.db.read_only or (scheduler_schema is not None and scheduler_schema[0] == 1):
                    self.idempotency = IdempotencyStore(self.db, self.instance_id)
                    self.scheduler = Scheduler(self.db, self.registry, self.root,
                        instance_id=self.instance_id, installation_id=self.db.installation_id,
                        runtime_profile=selection.profile, app_version=__version__,
                        capabilities=self.capabilities, instance_lease=self.lease)
                    self.scheduler.recover()
            self.preflight_receipt = self._make_preflight()
            self.auth = AuthManager(self.instance_id)
            self.initial_bootstrap = self.auth.issue_bootstrap()
            self.control = ControlServer(self.root, self.instance_id, os.getpid(), secrets.token_urlsafe(32), {
                'pair': self._control_pair, 'status': self._control_status,
                'submit_job': lambda args: self._audit_control('control_submit_job', self._control_submit, args),
                'cancel_job': lambda args: self._audit_control('control_cancel_job', self._control_cancel, args),
                'list_events': self._control_events,
            })
            self.control.start()
            if self.scheduler is not None and self.storage_writable:
                self._thread = threading.Thread(target=self._schedule, name='llmf-scheduler', daemon=True)
                self._thread.start()
        except (DatabaseError, StorageSecurityError) as exc:
            self.close()
            raise storage_error(exc) from None
        except BaseException:
            self.close()
            raise

    @property
    def storage_writable(self):
        return self.db is not None and not self.db.read_only

    def _require_registry(self):
        if self.registry is None:
            raise ApiError('STORAGE_UNAVAILABLE', 'The registry is unavailable in recovery mode.')
        return self.registry

    def _require_scheduler(self):
        if self.scheduler is None:
            raise ApiError('STORAGE_UNAVAILABLE', 'Jobs are unavailable in recovery mode.')
        return self.scheduler

    def _require_write(self):
        if not self.storage_writable:
            raise ApiError('STORAGE_UNAVAILABLE', 'Storage is in read-only recovery mode.')

    def _storage_probe(self):
        if not self.storage_writable or self.registry is None:
            return StorageProbeResult(False, False, False, False, 'Storage is unavailable.', 'Storage is unavailable.', 'Storage is unavailable.')
        staging = None
        wrote = read = deleted = False
        try:
            staging = self.registry.create_staging_dir(str(uuid.uuid4()))
            target = staging / 'preflight-probe'
            raw = secrets.token_bytes(4096)
            with target.open('xb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            wrote = True
            read = hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(raw).digest()
            target.unlink()
            deleted = not target.exists()
        except (OSError, ApiError, DatabaseError, StorageSecurityError):
            pass
        finally:
            if staging is not None:
                try:
                    self._remove_staging(staging)
                except OSError:
                    deleted = False
        if not (wrote and read and deleted):
            self.db.enter_read_only_recovery('STORAGE_PROBE_FAILED')
        return StorageProbeResult(wrote and read and deleted, wrote, read, deleted,
            '4096 random bytes written and flushed.' if wrote else 'Storage write failed.',
            'Read digest matched.' if read else 'Storage read or digest check failed.',
            'Probe deleted and absence verified.' if deleted else 'Storage deletion failed.')

    def _make_preflight(self):
        probe = self._storage_probe()
        artifact_id = self.registry.allocate_artifact_id() if self.storage_writable and self.registry is not None else None
        receipt = build_preflight_receipt(self.selection, self.child, probe,
            storage_root_display=self.root.name or 'storage', started_at=self.started_at,
            finished_at=utc_now(), gpu_query=self.query, receipt_artifact_id=artifact_id)
        validate_schema({'$ref': '#/components/schemas/PreflightReceipt'}, receipt, document='openapi.json')
        if artifact_id is not None:
            try:
                self.registry.register_stream(io.BytesIO(canonical_json(receipt)),
                    staging_owner=str(uuid.uuid4()), artifact_type='preflight_receipt',
                    display_name='Startup preflight', media_type='application/json',
                    preview_policy='text', origin='locally_created', artifact_id=artifact_id,
                    max_bytes=1048576)
            except (OSError, ApiError, DatabaseError, StorageSecurityError):
                self.db.enter_read_only_recovery('PREFLIGHT_REGISTRATION_FAILED')
                failed = StorageProbeResult(False, probe.write_ok, probe.read_digest_ok, False,
                    probe.write_observed, probe.read_observed, 'The receipt could not be registered.')
                receipt = build_preflight_receipt(self.selection, self.child, failed,
                    storage_root_display=self.root.name or 'storage', started_at=self.started_at,
                    finished_at=utc_now(), gpu_query=self.query)
        return receipt

    def runtime_info(self):
        return {'mode': 'local', 'api_version': '1.0', 'runtime_version': __version__,
            'instance_id': self.instance_id, 'profile': self.selection.profile,
            'device': self.selection.device, 'storage_root_display': self.root.name or 'storage',
            'storage_writable': self.storage_writable, 'server_time': utc_now(),
            'limits': dict(LIMITS), 'capabilities': self.capabilities, 'gpu_offer': self.gpu_offer}

    def _schedule(self):
        while not self._stop.wait(0.05):
            try:
                if self.storage_writable:
                    self.scheduler.tick()
            except (ApiError, OSError, DatabaseError, sqlite3.DatabaseError):
                self.db.enter_read_only_recovery('SCHEDULER_STORAGE_FAILURE')

    @staticmethod
    def _control_args(arguments, properties, required):
        return validate_schema({'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False}, arguments, document='openapi.json')

    def _control_pair(self, arguments):
        arguments = self._control_args(arguments, {}, [])
        return self.auth.issue_bootstrap()

    def _control_status(self, arguments):
        arguments = self._control_args(arguments, {}, [])
        return {**self.runtime_info(), 'pid': os.getpid()}

    def _audit_control(self, operation, action, arguments):
        request_id = str(uuid.uuid4())
        outcome, entity_ids = 'failed', ()
        try:
            result = action(arguments)
            outcome = 'accepted'
            if isinstance(result, dict) and isinstance(result.get('job_id'), str):
                entity_ids = (result['job_id'],)
            return result
        finally:
            self.audit_request(request_id, None, operation, outcome, entity_ids)

    def _control_submit(self, arguments):
        arguments = self._control_args(arguments, {'request': {'$ref': '#/components/schemas/JobRequest'}, 'idempotency_key': {'$ref': '#/components/schemas/Identifier'}}, ['request', 'idempotency_key'])
        _, body, _ = self.submit_job(arguments['request'], arguments['idempotency_key'])
        return strict_json(body)

    def _control_cancel(self, arguments):
        arguments = self._control_args(arguments, {'job_id': {'$ref': '#/components/schemas/Identifier'}}, ['job_id'])
        return self.cancel_job(arguments['job_id'])

    def _control_events(self, arguments):
        arguments = self._control_args(arguments, {'job_id': {'$ref': '#/components/schemas/Identifier'}, 'after_cursor': {'type': 'integer', 'minimum': 0, 'maximum': 2147483647}, 'limit': {'type': 'integer', 'minimum': 1, 'maximum': 500}}, ['job_id', 'after_cursor'])
        return self.list_events(arguments['job_id'], arguments['after_cursor'], arguments.get('limit', 100))

    def submit_job(self, request, key):
        self._require_write()
        result = self._require_scheduler().submit(request, key)
        return result.status_code, result.response_body, result.replayed

    def list_jobs(self, params):
        return self._require_scheduler().list(**params)

    def get_job(self, job_id):
        return self._require_scheduler().get(job_id)

    def cancel_job(self, job_id):
        self._require_write()
        return self._require_scheduler().cancel(job_id)

    def list_events(self, job_id, after_cursor, limit):
        return self._require_scheduler().list_events(job_id, after_cursor=after_cursor, limit=limit).as_dict()

    def list_registry(self, kind, params):
        params = dict(params)
        limit, cursor = params.pop('limit', 50), params.pop('cursor', None)
        return self._require_registry().list_records(kind, limit=limit, cursor=cursor, filters=params)

    def list_datasets(self, params):
        self._require_registry()
        return self.datasets.list(**params)

    def get_dataset(self, dataset_id):
        self._require_registry()
        return self.datasets.get(dataset_id)

    def get_artifact(self, artifact_id):
        return self._require_registry().get_artifact(artifact_id)

    def list_artifacts(self, params):
        return self._require_registry().list_artifacts(**params)

    def open_artifact(self, artifact_id):
        registry = self._require_registry()
        descriptor = registry.get_artifact(artifact_id)
        return descriptor, registry.open_verified_artifact(artifact_id)

    def begin_upload(self, length, route):
        self._require_write()
        registry = self._require_registry()
        owner = str(uuid.uuid4())
        rows = {'artifact_rows': 5, 'dataset_rows': 1} if route == '/api/v1/datasets' else {'artifact_rows': 1}
        registry.reserve_capacity(owner_kind='upload', owner_id=owner, byte_count=length + 64 * 1024 * 1024, reservation_id=owner, **rows)
        try:
            return owner, registry.create_staging_dir(owner)
        except BaseException:
            registry.release_capacity_now(owner)
            raise

    def register_upload(self, route, staged, key, owner):
        from .scheduler import IdempotencyCommit
        self._require_write()
        with self._upload_lock:
            existing = self.idempotency.lookup(key=key, request_sha256=staged.canonical_sha256, method='POST', resolved_path=route)
            if existing is not None:
                return existing.status_code, existing.response_body, True
            commit = IdempotencyCommit(self.idempotency, method='POST', resolved_path=route, key=key, request_sha256=staged.canonical_sha256)
            if route == '/api/v1/datasets':
                with contextlib.ExitStack() as stack:
                    parts = {p.name: stack.enter_context(p.path.open('rb')) for p in staged.parts}
                    result = self.datasets.register(staged.metadata, parts, upload_id=owner, idempotency_commit=commit)
            else:
                part = staged.part('bundle')
                inspect_zip_archive(part.path)
                with part.path.open('rb') as stream:
                    result = self.registry.register_stream(stream, staging_owner=owner,
                        artifact_type='bundle', display_name='Uploaded bundle', media_type='application/vnd.llm-foundations.bundle+zip',
                        preview_policy='download_only', origin='imported', max_bytes=1073741824,
                        expected_sha256=part.sha256, idempotency_commit=commit)
            return 201, canonical_json(result), False

    def _remove_staging(self, staging):
        path = Path(staging)
        expected = self.root / 'jobs'
        if path.name != 'staging' or path.parent.parent != expected or path.is_symlink():
            raise RuntimeError('Invalid internal staging cleanup target')
        uuid.UUID(path.parent.name)
        # Only the newly allocated, service-owned staging tree is removed.
        if path.exists():
            shutil.rmtree(path)
        with contextlib.suppress(OSError):
            path.parent.rmdir()

    def end_upload(self, owner, staging):
        try:
            self._remove_staging(staging)
        finally:
            try:
                self.registry.release_capacity_now(owner)
            except (DatabaseError, OSError, sqlite3.DatabaseError):
                self.db.enter_read_only_recovery('RESERVATION_RELEASE_FAILED')

    def revoke_session(self, identity, key):
        self._require_write()
        digest = hashlib.sha256(canonical_json({})).hexdigest()
        with self.db.transaction() as conn:
            prior = self.idempotency.lookup_in(conn, 'DELETE', '/api/v1/sessions/current', key, digest)
            if prior is not None:
                return prior.status_code, prior.response_body, True
            self.idempotency.record_success(conn, method='DELETE', resolved_path='/api/v1/sessions/current', key=key, request_sha256=digest, status_code=204, response_body=b'')
        self.auth.revoke_identity(identity)
        return 204, b'', False

    def audit_request(self, request_id, identity, operation, outcome, entity_ids=()):
        if not self.storage_writable:
            return
        session_hash = hashlib.sha256((identity.session_id if identity else 'unauthenticated').encode()).hexdigest()
        try:
            self.db.audit_mutation(request_id=request_id, session_hash=session_hash, operation=operation,
                entity_ids=entity_ids, outcome=outcome)
        except (DatabaseError, OSError, sqlite3.DatabaseError):
            self.db.enter_read_only_recovery('AUDIT_WRITE_FAILED')

    def close(self):
        if self._closed:
            return
        self._stop.set()
        control_error = None
        if self.control is not None:
            try:
                self.control.stop()
            except Exception as exc:
                control_error = exc
        if self._thread is not None:
            self._thread.join(timeout=5)
        try:
            if self.scheduler is not None:
                self.scheduler.shutdown()
        finally:
            if hasattr(self, 'auth'):
                self.auth.restart()
        # A failed owned-worker shutdown deliberately retains the database and
        # lease. A subsequent close can retry; ownership must precede release.
        try:
            if self.db is not None:
                self.db.close()
        finally:
            if self.lease is not None:
                self.lease.release()
        self._closed = True
        if control_error is not None:
            raise control_error

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
