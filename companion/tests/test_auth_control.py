from __future__ import annotations

import json
import os
import socket
import stat
from pathlib import Path

import pytest

from llm_foundations_companion.auth import (
    EXPECTED_HOST,
    EXPECTED_ORIGIN,
    MAX_SESSIONS,
    SESSION_ABSOLUTE_SECONDS,
    SESSION_IDLE_SECONDS,
    AuthManager,
    generate_instance_id,
)
from llm_foundations_companion.control import (
    MAX_CONTROL_LINE_BYTES,
    ControlClient,
    ControlServer,
    _decode_control_response,
)
from llm_foundations_companion.errors import ApiError
from llm_foundations_companion.schema import canonical_json, strict_json


class FakeClock:
    def __init__(self, now: float = 1_800_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _exchange(manager: AuthManager):
    grant = manager.issue_bootstrap()
    return manager.exchange_bootstrap(grant.secret)


def _mutation_headers(credentials) -> dict[str, str]:
    return {
        "Host": EXPECTED_HOST,
        "Origin": EXPECTED_ORIGIN,
        "Authorization": f"Bearer {credentials.access_token}",
        "X-LLMF-CSRF": credentials.csrf_token,
    }


def _assert_error(
    error: pytest.ExceptionInfo[ApiError],
    code: str,
    status: int,
    reason: str | None = None,
) -> None:
    assert error.value.code == code
    assert error.value.status_code == status
    assert error.value.reason_code == reason


def test_bootstrap_is_single_use_replaced_and_expires_at_120_seconds() -> None:
    clock = FakeClock()
    manager = AuthManager(generate_instance_id(), clock=clock)

    superseded = manager.issue_bootstrap()
    current = manager.issue_bootstrap()
    assert len(current.secret) == 43
    assert current.secret not in repr(current)
    assert current.secret not in repr(manager)

    with pytest.raises(ApiError) as stale:
        manager.exchange_bootstrap(superseded.secret)
    _assert_error(stale, "AUTH_REQUIRED", 401)

    credentials = manager.exchange_bootstrap(current.secret)
    assert credentials.as_dict() == {
        "access_token": credentials.access_token,
        "csrf_token": credentials.csrf_token,
        "instance_id": manager.instance_id,
        "idle_expires_at": "2027-01-15T10:00:00.000Z",
        "absolute_expires_at": "2027-01-15T20:00:00.000Z",
    }
    assert credentials.access_token not in repr(credentials)
    assert credentials.csrf_token not in repr(credentials)
    with pytest.raises(ApiError) as reused:
        manager.exchange_bootstrap(current.secret)
    _assert_error(reused, "AUTH_REQUIRED", 401)

    expiring = manager.issue_bootstrap()
    clock.advance(120)
    with pytest.raises(ApiError) as expired:
        manager.exchange_bootstrap(expiring.secret)
    _assert_error(expired, "AUTH_REQUIRED", 401)


def test_session_idle_absolute_revoke_and_restart_boundaries() -> None:
    clock = FakeClock()
    manager = AuthManager(generate_instance_id(), clock=clock)
    credentials = _exchange(manager)
    headers = _mutation_headers(credentials)

    clock.advance(SESSION_IDLE_SECONDS)
    with pytest.raises(ApiError) as idle_expired:
        manager.authorize("POST", headers)
    _assert_error(idle_expired, "SESSION_EXPIRED", 401)

    credentials = _exchange(manager)
    headers = _mutation_headers(credentials)
    created_at = clock.now
    while clock.now + (SESSION_IDLE_SECONDS - 1) < created_at + SESSION_ABSOLUTE_SECONDS:
        clock.advance(SESSION_IDLE_SECONDS - 1)
        manager.authorize("POST", headers)
    clock.now = created_at + SESSION_ABSOLUTE_SECONDS
    with pytest.raises(ApiError) as absolute_expired:
        manager.authorize("POST", headers)
    _assert_error(absolute_expired, "SESSION_EXPIRED", 401)

    credentials = _exchange(manager)
    headers = _mutation_headers(credentials)
    identity = manager.authorize("POST", headers)
    assert identity is not None
    assert identity.session_id == credentials.session_id
    assert credentials.access_token not in repr(identity)
    assert manager.revoke_identity(identity)
    with pytest.raises(ApiError) as revoked:
        manager.authorize("POST", headers)
    _assert_error(revoked, "AUTH_REQUIRED", 401)

    credentials = _exchange(manager)
    headers = _mutation_headers(credentials)
    manager.restart(generate_instance_id())
    with pytest.raises(ApiError) as restarted:
        manager.authorize("POST", headers)
    _assert_error(restarted, "AUTH_REQUIRED", 401)


def test_nonrefreshing_stream_check_does_not_extend_idle_expiry() -> None:
    clock = FakeClock()
    manager = AuthManager(generate_instance_id(), clock=clock)
    credentials = _exchange(manager)
    headers = _mutation_headers(credentials)

    clock.advance(SESSION_IDLE_SECONDS - 1)
    manager.authorize("POST", headers, refresh_idle=False)
    clock.advance(2)
    with pytest.raises(ApiError) as expired:
        manager.authorize("POST", headers)
    _assert_error(expired, "SESSION_EXPIRED", 401)


def test_session_limit_counts_only_unexpired_sessions_and_does_not_consume_grant() -> None:
    clock = FakeClock()
    manager = AuthManager(generate_instance_id(), clock=clock)
    sessions = [_exchange(manager) for _ in range(MAX_SESSIONS)]
    blocked_grant = manager.issue_bootstrap()

    with pytest.raises(ApiError) as limited:
        manager.exchange_bootstrap(blocked_grant.secret)
    _assert_error(limited, "REGISTRY_LIMIT", 409, "SESSION_LIMIT")

    assert manager.revoke(sessions[0].access_token)
    replacement = manager.exchange_bootstrap(blocked_grant.secret)
    assert replacement.instance_id == manager.instance_id
    assert manager.active_session_count() == MAX_SESSIONS


@pytest.mark.parametrize(
    ("method", "headers", "code", "status", "reason"),
    [
        (
            "POST",
            {"Host": "localhost:8765"},
            "ORIGIN_REJECTED",
            403,
            "HOST_MISMATCH",
        ),
        (
            "OPTIONS",
            {"Host": EXPECTED_HOST, "Origin": "http://evil.example"},
            "ORIGIN_REJECTED",
            403,
            "PREFLIGHT_REJECTED",
        ),
        (
            "POST",
            {"Host": EXPECTED_HOST},
            "ORIGIN_REJECTED",
            403,
            "ORIGIN_MISSING",
        ),
        (
            "POST",
            {"Host": EXPECTED_HOST, "Origin": "http://evil.example"},
            "ORIGIN_REJECTED",
            403,
            "ORIGIN_MISMATCH",
        ),
        (
            "GET",
            {"Host": EXPECTED_HOST, "Sec-Fetch-Site": "cross-site"},
            "ORIGIN_REJECTED",
            403,
            "FETCH_SITE_REJECTED",
        ),
        (
            "GET",
            {"Host": EXPECTED_HOST},
            "ORIGIN_REJECTED",
            403,
            "FETCH_SITE_REJECTED",
        ),
        (
            "POST",
            {"Host": EXPECTED_HOST, "Origin": EXPECTED_ORIGIN},
            "AUTH_REQUIRED",
            401,
            None,
        ),
    ],
)
def test_request_boundary_rejects_at_first_failed_stage(
    method: str,
    headers: dict[str, str],
    code: str,
    status: int,
    reason: str | None,
) -> None:
    manager = AuthManager(generate_instance_id())
    with pytest.raises(ApiError) as caught:
        manager.authorize(method, headers)
    _assert_error(caught, code, status, reason)


def test_request_boundary_expiry_precedes_csrf_and_csrf_precedes_http_body_stages() -> None:
    clock = FakeClock()
    manager = AuthManager(generate_instance_id(), clock=clock)
    credentials = _exchange(manager)
    expired_headers = _mutation_headers(credentials)
    expired_headers["X-LLMF-CSRF"] = "wrong"
    clock.advance(SESSION_IDLE_SECONDS)
    with pytest.raises(ApiError) as expired:
        manager.authorize("POST", expired_headers)
    _assert_error(expired, "SESSION_EXPIRED", 401)

    credentials = _exchange(manager)
    missing_csrf = _mutation_headers(credentials)
    missing_csrf.pop("X-LLMF-CSRF")
    with pytest.raises(ApiError) as csrf:
        manager.authorize("POST", missing_csrf)
    _assert_error(csrf, "CSRF_REJECTED", 403)

    # Content type, declared/streamed size, and body parsing are deliberately
    # later HTTP-facade stages.  A valid auth boundary returns before them.
    identity = manager.authorize("POST", _mutation_headers(credentials))
    assert identity is not None


def test_session_exchange_stops_after_origin_and_never_requires_bearer_or_csrf() -> None:
    manager = AuthManager(generate_instance_id())
    assert (
        manager.authorize(
            "POST",
            {"Host": EXPECTED_HOST, "Origin": EXPECTED_ORIGIN},
            session_exchange=True,
        )
        is None
    )


def test_control_server_validates_instance_and_stop_is_safe_before_start(
    tmp_path: Path,
) -> None:
    instance_id = generate_instance_id()
    server = ControlServer(
        tmp_path,
        instance_id,
        os.getpid(),
        "c" * 43,
        {},
    )
    server.stop()
    server.stop()

    with pytest.raises(ValueError, match="UUIDv4"):
        ControlServer(tmp_path, "not-an-instance", os.getpid(), "c" * 43, {})


def test_control_server_cleans_transport_after_partial_prepare_failure(
    tmp_path: Path,
) -> None:
    blocked_control_path = tmp_path / "runtime" / "control.json"
    blocked_control_path.mkdir(parents=True)
    server = ControlServer(
        tmp_path,
        generate_instance_id(),
        os.getpid(),
        "c" * 43,
        {},
    )
    try:
        with pytest.raises((OSError, RuntimeError)):
            server.start()
    finally:
        server.stop()
    if os.name != "nt":
        assert not (tmp_path / "runtime" / "control.sock").exists()


@pytest.mark.skipif(os.name == "nt", reason="Unix-domain socket boundary")
def test_control_server_rejects_overlong_unix_socket_path(tmp_path: Path) -> None:
    root = tmp_path / "long-root"
    while len(os.fsencode(str(root / "runtime" / "control.sock"))) <= 107:
        root /= "x" * 32
    root.mkdir(parents=True)
    server = ControlServer(
        root,
        generate_instance_id(),
        os.getpid(),
        "c" * 43,
        {},
    )

    try:
        with pytest.raises(ApiError) as caught:
            server.start()
        _assert_error(caught, "STORAGE_UNAVAILABLE", 503)
        assert "shorter root" in caught.value.message
    finally:
        server.stop()

    assert not (root / "runtime" / "control.json").exists()
    assert not (root / "runtime" / "control.sock").exists()


def test_control_client_rejects_an_invalid_error_envelope() -> None:
    error = ApiError("AUTH_REQUIRED").as_error(generate_instance_id())
    error["retryable"] = not error["retryable"]
    with pytest.raises(ApiError) as caught:
        _decode_control_response(canonical_json({"ok": False, "error": error}))
    _assert_error(caught, "INVALID_REQUEST", 400)


def _raw_control(endpoint: str, payload: bytes) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(3.0)
        connection.connect(endpoint)
        connection.sendall(payload)
        connection.shutdown(socket.SHUT_WR)
        received = bytearray()
        while b"\n" not in received:
            chunk = connection.recv(65536)
            if not chunk:
                break
            received.extend(chunk)
    assert received.endswith(b"\n")
    assert received.count(b"\n") == 1
    return strict_json(bytes(received[:-1]))


@pytest.mark.skipif(os.name == "nt", reason="Unix-domain socket boundary")
def test_actual_unix_control_channel_modes_dispatch_and_secret_privacy(
    tmp_path: Path,
) -> None:
    instance_id = generate_instance_id()
    secret = "c" * 43
    auth = AuthManager(instance_id)
    handlers = {
        "status": lambda _: {"instance_id": instance_id, "pid": os.getpid()},
        "pair": lambda _: auth.issue_bootstrap(),
    }
    server = ControlServer(tmp_path, instance_id, os.getpid(), secret, handlers)
    try:
        descriptor = server.start()
        assert descriptor.transport == "unix"
        assert secret not in repr(descriptor)
        assert stat.S_IMODE((tmp_path / "runtime").stat().st_mode) == 0o700
        assert stat.S_IMODE((tmp_path / "runtime" / "control.json").stat().st_mode) == 0o600
        assert stat.S_IMODE((tmp_path / "runtime" / "control.sock").stat().st_mode) == 0o600

        client = ControlClient(tmp_path)
        status = client.verify_instance()
        assert status == {"instance_id": instance_id, "pid": os.getpid()}
        pair = client.pair()
        assert pair["url"].startswith(f"{EXPECTED_ORIGIN}/#/connect/")
        assert secret not in json.dumps(status)
        assert secret not in json.dumps(pair)

        with pytest.raises(ApiError) as unavailable:
            client.call("cancel_job", {"job_id": generate_instance_id()})
        _assert_error(unavailable, "CAPABILITY_UNAVAILABLE", 503)
    finally:
        server.stop()

    assert not (tmp_path / "runtime" / "control.json").exists()
    assert not (tmp_path / "runtime" / "control.sock").exists()


@pytest.mark.skipif(os.name == "nt", reason="Unix-domain socket boundary")
def test_control_channel_rejects_wrong_secret_oversize_multiline_and_malformed(
    tmp_path: Path,
) -> None:
    instance_id = generate_instance_id()
    secret = "s" * 43
    server = ControlServer(
        tmp_path,
        instance_id,
        os.getpid(),
        secret,
        {"status": lambda _: {"instance_id": instance_id, "pid": os.getpid()}},
    )
    try:
        endpoint = server.start().endpoint
        wrong = canonical_json(
            {"control_secret": "x" * 43, "operation": "status", "arguments": {}}
        ) + b"\n"
        response = _raw_control(endpoint, wrong)
        assert response["error"]["code"] == "AUTH_REQUIRED"
        assert secret not in json.dumps(response)

        response = _raw_control(endpoint, b"x" * (MAX_CONTROL_LINE_BYTES + 1) + b"\n")
        assert response["error"]["code"] == "PAYLOAD_TOO_LARGE"

        valid = canonical_json(
            {"control_secret": secret, "operation": "status", "arguments": {}}
        )
        response = _raw_control(endpoint, valid + b"\n" + valid + b"\n")
        assert response["error"]["code"] == "INVALID_REQUEST"

        response = _raw_control(endpoint, b"{not-json}\n")
        assert response["error"]["code"] == "INVALID_REQUEST"

        noncanonical = json.dumps(
            {"control_secret": secret, "operation": "status", "arguments": {}},
            sort_keys=True,
        ).encode("utf-8")
        response = _raw_control(endpoint, noncanonical + b"\n")
        assert response["error"]["code"] == "INVALID_REQUEST"

        unknown = canonical_json(
            {"control_secret": secret, "operation": "shutdown", "arguments": {}}
        ) + b"\n"
        response = _raw_control(endpoint, unknown)
        assert response["error"]["code"] == "INVALID_REQUEST"
        assert secret not in json.dumps(response)
    finally:
        server.stop()


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows named pipes")
def test_actual_windows_named_pipe_has_no_tcp_fallback(tmp_path: Path) -> None:
    instance_id = generate_instance_id()
    server = ControlServer(
        tmp_path,
        instance_id,
        os.getpid(),
        "w" * 43,
        {"status": lambda _: {"instance_id": instance_id, "pid": os.getpid()}},
    )
    try:
        descriptor = server.start()
        assert descriptor.transport == "named_pipe"
        assert descriptor.endpoint == rf"\\.\pipe\llm-foundations-{instance_id}"
        assert ControlClient(tmp_path).verify_instance()["instance_id"] == instance_id
    finally:
        server.stop()
