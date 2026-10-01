"""Owner commands use private IPC; only serve binds the fixed loopback port."""
from __future__ import annotations

import argparse
import os
import socket
import sys
import uuid
from pathlib import Path

from .errors import ApiError
from .preflight import PreflightError
from .schema import canonical_json, strict_json, validate_schema


def parser():
    result = argparse.ArgumentParser(prog='python -m llm_foundations_companion')
    commands = result.add_subparsers(dest='command', required=True)
    for name in ('serve', 'connect', 'status', 'request', 'cancel', 'events'):
        command = commands.add_parser(name)
        command.add_argument('--storage', type=Path, default=Path.home() / '.llm-foundations')
        if name == 'serve':
            command.add_argument('--profile', choices=('win-cpu', 'win-cuda', 'wsl-cpu', 'wsl-cuda'))
        if name == 'request':
            command.add_argument('--file', type=Path, required=True)
            command.add_argument('--idempotency-key', default=None)
        if name in {'cancel', 'events'}:
            command.add_argument('--job-id', required=True)
        if name == 'events':
            command.add_argument('--after', required=True)
    commands.add_parser('gpu-check')
    return result


def print_json(value, *, stream=sys.stdout):
    print(canonical_json(value).decode('utf-8'), file=stream, flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'gpu-check':
            from .preflight import gpu_check
            offer, sentence = gpu_check()
            print_json(offer)
            print(sentence)
            return 0
        if args.command == 'serve':
            return serve(args)
        from .control import ControlClient
        client = ControlClient(args.storage)
        if args.command == 'connect':
            print(client.pair()['url'], flush=True)
            return 0
        client.verify_instance()
        if args.command == 'status':
            response = client.call('status', {})
        elif args.command == 'request':
            with args.file.open('rb') as source:
                raw = source.read(1048577)
            if len(raw) > 1048576:
                raise ApiError('PAYLOAD_TOO_LARGE')
            request = strict_json(raw)
            request = validate_schema({'$ref': '#/components/schemas/JobRequest'}, request, document='openapi.json')
            key = args.idempotency_key or str(uuid.uuid4())
            validate_schema({'$ref': '#/components/schemas/Identifier'}, key, document='openapi.json')
            response = client.call('submit_job', {'request': request, 'idempotency_key': key})
        elif args.command == 'cancel':
            response = client.call('cancel_job', {'job_id': args.job_id})
        else:
            if not args.after.isascii() or not args.after.isdecimal() or len(args.after) > 10:
                raise ApiError('VALIDATION_FAILED', reason_code='SCHEMA_INVALID', field_errors=[{'field_path': 'after_cursor', 'message': 'Use an ASCII decimal cursor.'}])
            response = client.call('list_events', {'job_id': args.job_id, 'after_cursor': int(args.after)})
        print_json(response)
        return 0
    except PreflightError as exc:
        print(exc.reason_code + ": " + exc.message, file=sys.stderr)
        return 1
    except ApiError as exc:
        print_json(exc.as_envelope(str(uuid.uuid4())), stream=sys.stderr)
        return 1
    except (OSError, ValueError):
        error = ApiError('STORAGE_UNAVAILABLE', 'The requested local service or file is unavailable.')
        print_json(error.as_envelope(str(uuid.uuid4())), stream=sys.stderr)
        return 1


def serve(args):
    from .preflight import ProfileSelectionError, select_profile, run_gpu_query, run_preflight_child, require_startup_allowed
    try:
        selection = select_profile(args.profile)
        query = run_gpu_query(selection.platform_kind)
        child = run_preflight_child(selection, interpreter=sys.executable)
        require_startup_allowed(selection, child, query)
    except PreflightError as exc:
        print(exc.reason_code + ": " + str(exc), file=sys.stderr)
        return 1
    from .service import Service
    # Exclusive fixed bind: no alternate interface or port is chosen.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if os.name == 'nt':
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        try:
            listener.bind(('127.0.0.1', 8765))
        except OSError:
            raise ApiError('STORAGE_IN_USE', 'The fixed loopback port is already in use or unavailable.') from None
        listener.listen(128)
        listener.setblocking(False)
        with Service(args.storage, selection, child, query) as service:
            from .api import Application
            import uvicorn
            print('Storage: ' + str(service.root), flush=True)
            print(service.initial_bootstrap.url, flush=True)
            config = uvicorn.Config(Application(service), host='127.0.0.1', port=8765, access_log=False, log_config=None, log_level='warning', server_header=False, date_header=False, timeout_keep_alive=5, h11_max_incomplete_event_size=16384)
            uvicorn.Server(config).run(sockets=[listener])
        return 0
    finally:
        listener.close()
