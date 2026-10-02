"""Small ASGI facade for the sealed v1 HTTP contract.

Authentication and transport checks precede parsing. Route and response schemas
come from the immutable package contract bundle, including unavailable routes.
"""
from __future__ import annotations

import asyncio
import mimetypes
import re
import sqlite3
import time
import uuid
from importlib.resources import files
from urllib.parse import parse_qsl

from .auth import EXPECTED_HOST
from .errors import ApiError
from .contract_data import ContractDataError
from .database import DatabaseError
from .platform_security import StorageSecurityError
from .schema import canonical_json, load_document, strict_json, validate_schema
from .transport import MAX_MULTIPART_OVERHEAD_BYTES, receive_json, receive_multipart

CSP = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'; worker-src 'self'"
SECURITY_HEADERS = [(b'x-content-type-options', b'nosniff'), (b'referrer-policy', b'no-referrer'), (b'cross-origin-resource-policy', b'same-origin')]
JSON_LIMIT = 1048576


def invalid(field, message='The parameter is invalid.'):
    return ApiError('VALIDATION_FAILED', reason_code='SCHEMA_INVALID', field_errors=[{'field_path': field, 'message': message}])


def request_headers(scope):
    result = {}
    for key, raw in scope.get('headers', []):
        name, value = key.decode('ascii').lower(), raw.decode('latin-1')
        result[name] = result[name] + ',' + value if name in result else value
    return result


def content_length(headers, *, required=False):
    raw = headers.get('content-length')
    if raw is None:
        if required:
            raise invalid('Content-Length', 'An upload requires a known length.')
        return None
    if re.fullmatch(r'[0-9]{1,20}', raw) is None:
        raise invalid('Content-Length')
    value = int(raw)
    if value > (1 << 64) - 1:
        raise invalid('Content-Length')
    return value


def check_media_type(headers, expected):
    if headers.get('content-encoding') is not None:
        raise ApiError('UNSUPPORTED_MEDIA_TYPE', 'Encoded request bodies are not supported.')
    supplied = headers.get('content-type', '')
    pieces = [part.strip() for part in supplied.split(';')]
    if pieces[0].lower() != expected:
        raise ApiError('UNSUPPORTED_MEDIA_TYPE')
    if expected == 'application/json':
        if any(part.lower() not in {'charset=utf-8', 'charset="utf-8"'} for part in pieces[1:]):
            raise ApiError('UNSUPPORTED_MEDIA_TYPE')
    return supplied


class Application:
    def __init__(self, service):
        self.service = service
        self.contract = load_document('openapi.json')
        self.routes = []
        for template, methods in self.contract['paths'].items():
            pattern = '^' + re.sub(r'\\\{([a-z_]+)\\\}', r'(?P<\1>[^/]+)', re.escape(template)) + '$'
            self.routes.append((template, re.compile(pattern), methods))

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'lifespan':
            while True:
                message = await receive()
                if message['type'] == 'lifespan.startup':
                    await send({'type': 'lifespan.startup.complete'})
                elif message['type'] == 'lifespan.shutdown':
                    await send({'type': 'lifespan.shutdown.complete'})
                    return
        if scope['type'] != 'http':
            return
        request_id = str(uuid.uuid4())
        identity = None
        method, path = scope['method'].upper(), scope['path']
        mutation = method in {'POST', 'PUT', 'PATCH', 'DELETE'}
        response_started = False
        original_send = send
        async def tracked_send(message):
            nonlocal response_started
            if message['type'] == 'http.response.start':
                response_started = True
            await original_send(message)
        send = tracked_send
        outcome = 'failed'
        audit_operation = 'unknown_mutation'
        audit_entities = ()
        try:
            headers = request_headers(scope)
            if path.startswith('/api'):
                identity = self.service.auth.authorize(method, headers, session_exchange=(path == '/api/v1/sessions' and method == 'POST'))
            elif path != '/healthz' and headers.get('host') != EXPECTED_HOST:
                raise ApiError('ORIGIN_REJECTED', reason_code='HOST_MISMATCH')
            match = next(((template, regex.fullmatch(path), methods) for template, regex, methods in self.routes if regex.fullmatch(path)), None)
            if match is None:
                if path.startswith('/api'):
                    raise ApiError('NOT_FOUND')
                await self.static(method, path, send, request_id)
                return
            template, matched, methods = match
            operation = methods.get(method.lower())
            if operation is None:
                raise ApiError('METHOD_NOT_ALLOWED')
            audit_operation = operation.get('operationId', 'unknown_mutation')
            body_spec = operation.get('requestBody', {}).get('content', {})
            upload = 'multipart/form-data' in body_spec
            if body_spec:
                media = check_media_type(headers, 'multipart/form-data' if upload else 'application/json')
            elif headers.get('content-encoding'):
                raise ApiError('UNSUPPORTED_MEDIA_TYPE')
            length = content_length(headers, required=upload)
            maximum = (1073741824 + MAX_MULTIPART_OVERHEAD_BYTES if template == '/api/v1/bundle-uploads' else 31457280 + JSON_LIMIT + MAX_MULTIPART_OVERHEAD_BYTES) if upload else JSON_LIMIT
            if length is not None and length > maximum:
                raise ApiError('PAYLOAD_TOO_LARGE')
            parameters = self.parameters(operation, matched.groupdict(), scope.get('query_string', b''), headers)
            if upload:
                if template not in {'/api/v1/datasets', '/api/v1/bundle-uploads'}:
                    raise ApiError('CAPABILITY_UNAVAILABLE', 'This operation is not available in this work slice.')
                owner, staging = self.service.begin_upload(length, template)
                try:
                    staged = await receive_multipart(receive, content_type=media, content_length=length, staging_dir=staging, upload_kind='bundle' if template == '/api/v1/bundle-uploads' else 'dataset')
                    status, result, replayed = self.service.register_upload(template, staged, parameters['Idempotency-Key'], owner)
                finally:
                    self.service.end_upload(owner, staging)
            else:
                body = None
                if body_spec:
                    body = await receive_json(receive, content_length=length)
                    body = validate_schema(body_spec['application/json']['schema'], body, document='openapi.json',
                        include_semantic=template != '/api/v1/jobs')
                elif method != 'GET' or length:
                    body_bytes = await self.empty_body(receive, length)
                    if body_bytes:
                        raise ApiError('INVALID_REQUEST', 'This operation has no request body.')
                if template == '/api/v1/jobs/{id}/event-stream':
                    await self.event_stream(parameters, headers, receive, send, request_id)
                    return
                if template == '/api/v1/artifacts/{id}/content':
                    if 'range' in headers:
                        raise ApiError('RANGE_NOT_SATISFIABLE')
                    await self.artifact_content(parameters['id'], send, request_id)
                    return
                status, result, replayed = self.dispatch(template, method, parameters, body, identity)
            try:
                if str(status) not in operation['responses']:
                    raise ValueError('undeclared response status')
                response_spec = operation['responses'][str(status)].get('content', {}).get('application/json', {}).get('schema')
                if status == 204:
                    if result not in (None, b''):
                        raise ValueError('nonempty no-content response')
                    payload, raw = None, b''
                else:
                    if response_spec is None:
                        raise ValueError('missing JSON response contract')
                    payload = strict_json(result) if isinstance(result, bytes) else result
                    payload = validate_schema(response_spec, payload, document='openapi.json')
                    raw = canonical_json(payload)
                    if isinstance(result, bytes):
                        if raw != result:
                            raise ValueError('persisted response is not canonical')
                        raw = result
            except (ApiError, ValueError, ContractDataError):
                raise ApiError('INTERNAL_ERROR', 'The local response failed its contract check.') from None
            if isinstance(payload, dict):
                audit_entities = tuple(value for name, value in payload.items()
                    if name in {'dataset_id', 'artifact_id', 'job_id', 'run_id', 'model_id', 'checkpoint_id'}
                    and isinstance(value, str) and re.fullmatch(r'[0-9a-f-]{36}', value))
            extra = [(b'idempotent-replayed', b'true')] if replayed else []
            await self.send_bytes(send, status, raw, 'application/json', request_id, extra)
            outcome = 'accepted'
        except ApiError as exc:
            if response_started:
                await send({'type': 'http.response.body', 'body': b''})
                return
            await self.send_bytes(send, exc.status_code, canonical_json(exc.as_envelope(request_id)), 'application/json', request_id)
        except (DatabaseError, StorageSecurityError, sqlite3.DatabaseError):
            database = getattr(self.service, 'db', None)
            if database is not None:
                database.enter_read_only_recovery('HTTP_STORAGE_FAILURE')
            if response_started:
                await send({'type': 'http.response.body', 'body': b''})
                return
            error = ApiError('STORAGE_UNAVAILABLE', 'The storage root is unavailable or in recovery mode.')
            await self.send_bytes(send, error.status_code, canonical_json(error.as_envelope(request_id)), 'application/json', request_id)
        except (OSError, ValueError, RuntimeError):
            if response_started:
                await send({'type': 'http.response.body', 'body': b''})
                return
            error = ApiError('INTERNAL_ERROR', 'The local request could not be completed.')
            await self.send_bytes(send, error.status_code, canonical_json(error.as_envelope(request_id)), 'application/json', request_id)
        finally:
            if mutation and path.startswith('/api'):
                self.service.audit_request(request_id, identity, audit_operation, outcome, audit_entities)

    async def empty_body(self, receive, length):
        total = 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                raise ApiError('INVALID_REQUEST', 'The request was disconnected.')
            total += len(message.get('body', b''))
            if total > JSON_LIMIT:
                raise ApiError('PAYLOAD_TOO_LARGE')
            if not message.get('more_body', False):
                break
        if length is not None and total != length:
            raise invalid('Content-Length')
        return total

    def parameters(self, operation, path_values, raw_query, headers):
        try:
            query = parse_qsl(raw_query.decode('ascii'), keep_blank_values=True, strict_parsing=True, encoding='utf-8', errors='strict', max_num_fields=64)
        except (UnicodeError, ValueError):
            raise invalid('query') from None
        allowed = {p['name'] for p in operation.get('parameters', []) if p['in'] == 'query'}
        query_values = {}
        for name, value in query:
            if name not in allowed or name in query_values:
                raise invalid(name, 'Unknown or repeated query parameter.')
            query_values[name] = value
        values = {}
        for parameter in operation.get('parameters', []):
            name, location, schema = parameter['name'], parameter['in'], parameter['schema']
            if location == 'header' and name.lower() in {'origin', 'sec-fetch-site', 'x-llmf-csrf'}:
                continue  # Ordered authentication already checked these.
            source = path_values if location == 'path' else query_values if location == 'query' else headers
            key = name.lower() if location == 'header' else name
            if key not in source:
                if parameter.get('required'):
                    raise invalid(name, 'The parameter is required.')
                if 'default' in schema:
                    values[name] = schema['default']
                continue
            value = source[key]
            if schema.get('type') == 'integer':
                if re.fullmatch(r'[0-9]+', value) is None or len(value) > 20:
                    raise invalid(name, 'Use an ASCII decimal integer in the permitted range.')
                value = int(value)
            try:
                value = validate_schema(schema, value, document='openapi.json')
            except ApiError:
                raise invalid(name) from None
            values[name] = value
        return values

    def dispatch(self, path, method, params, body, identity):
        s = self.service
        if path == '/healthz':
            return 200, {'status': 'ok', 'api_version': '1.0'}, False
        if path == '/api/v1/sessions':
            return 201, s.auth.exchange_bootstrap(body['bootstrap_secret']).as_dict(), False
        if path == '/api/v1/sessions/current':
            return s.revoke_session(identity, params['Idempotency-Key'])
        if path == '/api/v1/runtime':
            return 200, s.runtime_info(), False
        if path == '/api/v1/preflight':
            return 200, s.preflight_receipt, False
        if path == '/api/v1/datasets':
            return 200, s.list_datasets(params), False
        if path == '/api/v1/datasets/{id}':
            return 200, s.get_dataset(params['id']), False
        if path == '/api/v1/jobs':
            if method == 'POST':
                return s.submit_job(body, params['Idempotency-Key'])
            return 200, s.list_jobs(params), False
        if path == '/api/v1/jobs/{id}':
            return 200, s.get_job(params['id']), False
        if path == '/api/v1/jobs/{id}/cancel':
            return 200, s.cancel_job(params['id']), False
        if path == '/api/v1/jobs/{id}/events':
            return 200, s.list_events(params['id'], params['after_cursor'], params.get('limit', 100)), False
        if path == '/api/v1/artifacts/{id}':
            return 200, s.get_artifact(params['id']), False
        if path == '/api/v1/artifacts':
            return 200, s.list_artifacts(params), False
        if method == 'GET' and path in {'/api/v1/runs', '/api/v1/models', '/api/v1/checkpoints', '/api/v1/tokenizers'}:
            return 200, s.list_registry(path.rsplit('/', 1)[-1], params), False
        if method == 'GET' and path in {'/api/v1/runs/{id}', '/api/v1/models/{id}', '/api/v1/checkpoints/{id}', '/api/v1/tokenizers/{id}'}:
            raise ApiError('NOT_FOUND')
        raise ApiError('CAPABILITY_UNAVAILABLE', 'This operation is not available in this work slice.')

    async def send_bytes(self, send, status, body, media, request_id, extra=()):
        headers = SECURITY_HEADERS + [(b'content-type', media.encode('ascii')), (b'content-length', str(len(body)).encode()), (b'x-request-id', request_id.encode()), (b'cache-control', b'no-store')] + list(extra)
        await send({'type': 'http.response.start', 'status': status, 'headers': headers})
        await send({'type': 'http.response.body', 'body': body})

    async def static(self, method, path, send, request_id):
        if method != 'GET':
            raise ApiError('METHOD_NOT_ALLOWED')
        parts = ('local-index.html' if path == '/' else path.lstrip('/')).split('/')
        if any(not p or p in {'.', '..'} or '\\' in p or ':' in p for p in parts):
            raise ApiError('NOT_FOUND')
        from .schema import strict_json
        root = files('llm_foundations_companion').joinpath('static')
        manifest = strict_json(root.joinpath('manifest.json').read_bytes())
        if '/'.join(parts) not in {row['path'] for row in manifest['files'] + manifest['authored_files']}:
            raise ApiError('NOT_FOUND')
        resource = root.joinpath(*parts)
        if not resource.is_file():
            raise ApiError('NOT_FOUND')
        media = mimetypes.guess_type(parts[-1])[0] or 'application/octet-stream'
        extra = [(b'content-security-policy', CSP.encode())] if media == 'text/html' else []
        await self.send_bytes(send, 200, resource.read_bytes(), media, request_id, extra)

    async def artifact_content(self, artifact_id, send, request_id):
        descriptor, stream = self.service.open_artifact(artifact_id)
        try:
            media = descriptor['media_type']
            preview = descriptor.get('preview_policy')
            inline = preview in {'text', 'json', 'inline_text'} and descriptor['size_bytes'] <= 262144 and media in {'text/plain', 'application/json', 'application/x-ndjson'}
            if inline:
                sample = stream.read()
                try:
                    sample.decode('utf-8', 'strict')
                except UnicodeError:
                    inline = False
                stream.seek(0)
            headers = SECURITY_HEADERS + [(b'content-type', (media if inline else 'application/octet-stream').encode('ascii')), (b'content-length', str(descriptor['size_bytes']).encode()), (b'content-disposition', ('inline' if inline else 'attachment; filename="artifact-' + artifact_id + '"').encode('ascii')), (b'x-request-id', request_id.encode()), (b'cache-control', b'no-store')]
            await send({'type': 'http.response.start', 'status': 200, 'headers': headers})
            while data := stream.read(65536):
                await send({'type': 'http.response.body', 'body': data, 'more_body': True})
            await send({'type': 'http.response.body', 'body': b''})
        finally:
            stream.close()

    @staticmethod
    def _stream_record(schema_name, value):
        try:
            return validate_schema({'$ref': '#/components/schemas/' + schema_name}, value, document='openapi.json')
        except (ApiError, ValueError, ContractDataError):
            raise ApiError('INTERNAL_ERROR', 'The event stream failed its contract check.') from None

    async def event_stream(self, params, headers, receive, send, request_id):
        cursor = params['after_cursor']
        page = self._stream_record('EventPage', self.service.list_events(params['id'], cursor, 500))
        self._stream_record('Job', self.service.get_job(params['id']))
        await send({'type': 'http.response.start', 'status': 200, 'headers': SECURITY_HEADERS + [(b'content-type', b'text/event-stream'), (b'cache-control', b'no-store'), (b'x-request-id', request_id.encode())]})
        deadline, heartbeat, sent = time.monotonic() + 300, time.monotonic() + 15, 0
        disconnected = asyncio.create_task(receive())
        try:
            while time.monotonic() < deadline and sent < 500:
                if disconnected.done():
                    if disconnected.result()['type'] == 'http.disconnect':
                        return
                    disconnected = asyncio.create_task(receive())
                try:
                    self.service.auth.authorize('GET', headers, refresh_idle=False)
                except ApiError:
                    break
                for event in page['items']:
                    frame = b'id: ' + str(event['cursor']).encode() + b'\nevent: ' + event['event_type'].encode('ascii') + b'\ndata: ' + canonical_json(event) + b'\n\n'
                    await send({'type': 'http.response.body', 'body': frame, 'more_body': True})
                    cursor, sent = event['cursor'], sent + 1
                    if sent == 500:
                        break
                if sent == 500:
                    break
                job = self._stream_record('Job', self.service.get_job(params['id']))
                if job['state'] in {'completed', 'failed', 'interrupted'} and cursor >= job['last_cursor']:
                    break
                if time.monotonic() >= heartbeat:
                    await send({'type': 'http.response.body', 'body': b': heartbeat\n\n', 'more_body': True})
                    heartbeat = time.monotonic() + 15
                await asyncio.sleep(0.1)
                page = self._stream_record('EventPage', self.service.list_events(params['id'], cursor, min(500 - sent, 500)))
            await send({'type': 'http.response.body', 'body': b''})
        finally:
            disconnected.cancel()
            try:
                await disconnected
            except asyncio.CancelledError:
                pass
