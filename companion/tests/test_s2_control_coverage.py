from __future__ import annotations

import ctypes
import json
import os
import socket
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

import llm_foundations_companion.auth as auth
import llm_foundations_companion.control as control
from llm_foundations_companion.errors import ApiError
from llm_foundations_companion.schema import canonical_json


_INSTANCE_ID = "12345678-1234-4234-9234-123456789abc"
_SECRET = "s" * 43


class _OSProxy:
    def __init__(self, name: str) -> None:
        self.name = name

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)


class _Call:
    def __init__(self, result: Any = 1, effect: Any = None) -> None:
        self.result = result
        self.effect = effect
        self.calls: list[tuple[Any, ...]] = []
        self.argtypes: Any = None
        self.restype: Any = None

    def __call__(self, *args: Any) -> Any:
        self.calls.append(args)
        if self.effect is not None:
            return self.effect(*args)
        if isinstance(self.result, list):
            return self.result.pop(0)
        return self.result


class _DLL:
    def __init__(self) -> None:
        self._calls: dict[str, _Call] = {}

    def __getattr__(self, name: str) -> _Call:
        return self._calls.setdefault(name, _Call())

    def set(self, name: str, call: _Call) -> _Call:
        self._calls[name] = call
        return call


class _DuplicateHeaders(Mapping[str, str]):
    def __getitem__(self, key: str) -> str:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self):
        return [
            ("Host", auth.EXPECTED_HOST),
            ("host", auth.EXPECTED_HOST),
            ("Sec-Fetch-Site", "same-origin"),
        ]


class _Serializable:
    def as_dict(self) -> dict[str, str]:
        return {"value": "safe"}


class _ScriptedSocket:
    def __init__(self, reads: list[Any], *, peek: bytes = b"") -> None:
        self.reads = list(reads)
        self.peek = peek
        self.blocking: list[bool] = []
        self.timeouts: list[float] = []

    def recv(self, _size: int, flags: int = 0) -> bytes:
        if flags:
            if isinstance(self.peek, BaseException):
                raise self.peek
            return self.peek
        item = self.reads.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def setblocking(self, value: bool) -> None:
        self.blocking.append(value)

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)


def _server(root: Path, handlers: Mapping[str, Any] | None = None) -> control.ControlServer:
    return control.ControlServer(
        root,
        _INSTANCE_ID,
        os.getpid(),
        _SECRET,
        handlers or {},
    )


def _request(operation: str, arguments: Mapping[str, Any] | None = None) -> bytes:
    return canonical_json(
        {
            "control_secret": _SECRET,
            "operation": operation,
            "arguments": dict(arguments or {}),
        }
    )


def _error_code(response: Mapping[str, Any]) -> str:
    assert response["ok"] is False
    return str(response["error"]["code"])


def _assert_api_code(caught: pytest.ExceptionInfo[ApiError], code: str) -> None:
    assert caught.value.code == code


def test_auth_helpers_fail_closed_without_retaining_credentials() -> None:
    with pytest.raises(RuntimeError, match="credential entropy"):
        auth._secret_from_bytes(b"short")
    with pytest.raises(RuntimeError, match="identifier entropy"):
        auth._uuid_from_bytes(b"short")
    assert not auth._canonical_uuid(None)  # type: ignore[arg-type]
    assert auth._bearer_token("Basic token") is None

    with pytest.raises(ValueError, match="canonical lowercase UUID"):
        auth.AuthManager("INVALID")

    manager = auth.AuthManager(_INSTANCE_ID, rng=lambda count: bytes(range(count)))
    credentials = manager.exchange_bootstrap(manager.issue_bootstrap().secret)
    assert not manager.revoke("not-a-session-token")

    with pytest.raises(ApiError) as duplicate_host:
        manager.authorize("GET", _DuplicateHeaders())
    assert duplicate_host.value.reason_code == "HOST_MISMATCH"

    with pytest.raises(ValueError, match="canonical lowercase UUID"):
        manager.restart("INVALID")
    with pytest.raises(ApiError) as cleared:
        manager.authorize(
            "POST",
            {
                "Host": auth.EXPECTED_HOST,
                "Origin": auth.EXPECTED_ORIGIN,
                "Authorization": f"Bearer {credentials.access_token}",
                "X-LLMF-CSRF": credentials.csrf_token,
            },
        )
    _assert_api_code(cleared, "AUTH_REQUIRED")


def test_control_constructor_request_and_dispatch_boundaries(tmp_path: Path) -> None:
    assert not control._canonical_instance_id(1)
    assert control._safe_result(_Serializable()) == {"value": "safe"}

    with pytest.raises(ValueError, match="pid"):
        control.ControlServer(tmp_path, _INSTANCE_ID, 0, _SECRET, {})
    with pytest.raises(ValueError, match="256 bits"):
        control.ControlServer(tmp_path, _INSTANCE_ID, os.getpid(), "short", {})

    valid = {
        "control_secret": _SECRET,
        "operation": "status",
        "arguments": {},
    }
    invalid_lines = [
        canonical_json(valid) + b" ",
        canonical_json([]),
        canonical_json({"control_secret": _SECRET, "operation": "status"}),
        canonical_json({**valid, "control_secret": 1}),
        canonical_json({**valid, "operation": 1}),
        canonical_json({**valid, "arguments": []}),
    ]
    for line in invalid_lines:
        with pytest.raises(ApiError) as caught:
            control._decode_request(line)
        _assert_api_code(caught, "INVALID_REQUEST")

    async def asynchronous(_arguments: Mapping[str, Any]) -> None:
        return None

    def failed(_arguments: Mapping[str, Any]) -> None:
        raise RuntimeError("do not disclose")

    server = _server(
        tmp_path,
        {
            "status": lambda _arguments: _Serializable(),
            "cancel_job": asynchronous,
            "list_events": failed,
        },
    )
    assert server.handle_line(_request("status")) == {
        "ok": True,
        "result": {"value": "safe"},
    }
    assert _error_code(server.handle_line(_request("pair"))) == "CAPABILITY_UNAVAILABLE"
    assert _error_code(server.handle_line(_request("cancel_job"))) == "CAPABILITY_UNAVAILABLE"
    assert _error_code(server.handle_line(_request("list_events"))) == "INTERNAL_ERROR"


def test_control_response_encoding_is_bounded_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = control.canonical_json

    def reject_unserializable(value: Any) -> bytes:
        if isinstance(value, dict) and value.get("unserializable") is True:
            raise TypeError("not serializable")
        return original(value)

    monkeypatch.setattr(control, "canonical_json", reject_unserializable)
    response = control._encode_response({"unserializable": True})
    decoded = json.loads(response)
    assert decoded["error"]["code"] == "INTERNAL_ERROR"

    oversized = control._encode_response(
        {"ok": True, "result": "x" * (control.MAX_CONTROL_LINE_BYTES + 1)}
    )
    assert json.loads(oversized)["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_control_line_reader_rejects_partial_multiple_and_queued_lines() -> None:
    line, failure = control._read_socket_line(_ScriptedSocket([b""]))
    assert line == b""
    assert _error_code(failure or {}) == "INVALID_REQUEST"

    line, failure = control._read_socket_line(_ScriptedSocket([TimeoutError()]))
    assert line == b""
    assert _error_code(failure or {}) == "INVALID_REQUEST"

    line, failure = control._read_socket_line(
        _ScriptedSocket([b"x" * (control.MAX_CONTROL_LINE_BYTES + 1), b""])
    )
    assert line == b""
    assert _error_code(failure or {}) == "PAYLOAD_TOO_LARGE"

    line, failure = control._read_socket_line(_ScriptedSocket([b"{}\n{}\n"]))
    assert line == b""
    assert _error_code(failure or {}) == "INVALID_REQUEST"

    queued = _ScriptedSocket([b"{}\n"], peek=b"x")
    line, failure = control._read_socket_line(queued)
    assert line == b""
    assert _error_code(failure or {}) == "INVALID_REQUEST"
    assert queued.blocking == [False, True]
    assert queued.timeouts == [2.0]

    clean = _ScriptedSocket([b"{}\n"])
    assert control._read_socket_line(clean) == (b"{}", None)


def test_control_descriptor_and_response_validation(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    path = runtime / "control.json"
    descriptor = control.ControlDescriptor(
        _INSTANCE_ID,
        os.getpid(),
        _SECRET,
        "unix",
        str(runtime / "control.sock"),
    )

    def write(raw: bytes) -> None:
        path.write_bytes(raw)
        path.chmod(0o600)

    write(canonical_json(descriptor.as_dict()) + b"\n")
    assert control._load_descriptor(path) == descriptor

    invalid = [
        b"x" * (control.MAX_CONTROL_LINE_BYTES + 1),
        b"{}",
        b"{not-json}\n",
        json.dumps(descriptor.as_dict(), sort_keys=True).encode() + b"\n",
        canonical_json({"instance_id": _INSTANCE_ID}) + b"\n",
        canonical_json({**descriptor.as_dict(), "pid": True}) + b"\n",
        canonical_json({**descriptor.as_dict(), "endpoint": "/tmp/other"}) + b"\n",
        canonical_json(
            {
                **descriptor.as_dict(),
                "transport": "named_pipe",
                "endpoint": r"\\.\pipe\wrong",
            }
        )
        + b"\n",
    ]
    for raw in invalid:
        write(raw)
        with pytest.raises(ApiError) as caught:
            control._load_descriptor(path)
        assert caught.value.code in {"INVALID_REQUEST", "PAYLOAD_TOO_LARGE"}

    assert control._decode_control_response(canonical_json({"ok": True, "result": 7})) == 7
    valid_error = ApiError(
        "VALIDATION_FAILED",
        reason_code="SCHEMA_INVALID",
        field_errors=[{"field_path": "x", "message": "bad"}],
    )
    envelope = canonical_json(
        {"ok": False, "error": valid_error.as_error(_INSTANCE_ID)}
    )
    with pytest.raises(ApiError) as decoded:
        control._decode_control_response(envelope)
    assert (decoded.value.code, decoded.value.reason_code) == (
        "VALIDATION_FAILED",
        "SCHEMA_INVALID",
    )

    for response in (
        b"{not-json}",
        canonical_json([]),
        canonical_json({"ok": True}),
        canonical_json({"ok": False, "error": {}}),
        canonical_json({"ok": "yes", "result": None}),
    ):
        with pytest.raises(ApiError) as caught:
            control._decode_control_response(response)
        _assert_api_code(caught, "INVALID_REQUEST")


def test_control_client_checks_operation_transport_identity_and_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = control.ControlClient(tmp_path)
    descriptor = control.ControlDescriptor(
        _INSTANCE_ID,
        os.getpid(),
        _SECRET,
        "unix",
        str(tmp_path / "control.sock"),
    )
    monkeypatch.setattr(
        client,
        "_exchange_unix",
        lambda _endpoint, _request: canonical_json({"ok": True, "result": {"safe": True}}),
    )
    assert client._call_with_descriptor(descriptor, "status", {}) == {"safe": True}

    with pytest.raises(ApiError) as unknown:
        client._call_with_descriptor(descriptor, "shutdown", {})
    _assert_api_code(unknown, "INVALID_REQUEST")

    with pytest.raises(ApiError) as oversized:
        client._call_with_descriptor(
            descriptor,
            "status",
            {"value": "x" * control.MAX_CONTROL_LINE_BYTES},
        )
    _assert_api_code(oversized, "PAYLOAD_TOO_LARGE")

    unavailable = control.ControlDescriptor(
        _INSTANCE_ID,
        os.getpid(),
        _SECRET,
        "named_pipe",
        rf"\\.\pipe\llm-foundations-{_INSTANCE_ID}",
    )
    with pytest.raises(ApiError) as wrong_platform:
        client._call_with_descriptor(unavailable, "status", {})
    _assert_api_code(wrong_platform, "CAPABILITY_UNAVAILABLE")

    monkeypatch.setattr(client, "descriptor", lambda: descriptor)
    monkeypatch.setattr(client, "_call_with_descriptor", lambda *_args: "invalid")
    with pytest.raises(ApiError) as invalid_status:
        client.verify_instance()
    _assert_api_code(invalid_status, "AUTH_REQUIRED")

    monkeypatch.setattr(
        client,
        "_call_with_descriptor",
        lambda _descriptor, operation, _arguments: (
            {"instance_id": _INSTANCE_ID, "pid": descriptor.pid}
            if operation == "status"
            else {"not_url": True}
        ),
    )
    with pytest.raises(ApiError) as invalid_pair:
        client.pair()
    _assert_api_code(invalid_pair, "INTERNAL_ERROR")

    monkeypatch.setattr(
        client,
        "_call_with_descriptor",
        lambda *_args: {"instance_id": _INSTANCE_ID, "pid": descriptor.pid + 1},
    )
    with pytest.raises(ApiError) as mismatch:
        client.verify_instance()
    _assert_api_code(mismatch, "AUTH_REQUIRED")


def test_unix_client_rejects_a_nonprivate_endpoint(tmp_path: Path) -> None:
    endpoint = tmp_path / "not-a-socket"
    endpoint.write_bytes(b"")
    endpoint.chmod(0o600)
    client = control.ControlClient(tmp_path)
    with pytest.raises(ApiError) as caught:
        client._exchange_unix(str(endpoint), b"{}\n")
    _assert_api_code(caught, "AUTH_REQUIRED")


@pytest.mark.skipif(os.name == "nt", reason="Unix-domain socket invariant")
def test_unix_server_replaces_stale_socket_and_preserves_replacements(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(runtime / "control.sock"))
    stale.close()

    server = _server(
        tmp_path,
        {"status": lambda _args: {"instance_id": _INSTANCE_ID, "pid": os.getpid()}},
    )
    replacement: socket.socket | None = None
    try:
        first = server.start()
        assert server.start() == first
        assert control.ControlClient(tmp_path).verify_instance()["instance_id"] == _INSTANCE_ID

        server.socket_path.unlink()
        replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        replacement.bind(str(server.socket_path))
        other = control.ControlDescriptor(
            "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            os.getpid(),
            "x" * 43,
            "unix",
            str(server.socket_path),
        )
        control.write_private_file(
            server.control_path,
            canonical_json(other.as_dict()) + b"\n",
        )
    finally:
        server.stop()

    assert server.control_path.exists()
    assert server.socket_path.exists()
    if replacement is not None:
        replacement.close()
    server.socket_path.unlink()
    server.control_path.unlink()


@pytest.mark.skipif(os.name == "nt", reason="Unix-domain socket invariant")
def test_unix_prepare_rejects_an_occupied_nonsocket(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    endpoint = runtime / "control.sock"
    endpoint.write_text("preserve", encoding="utf-8")
    server = _server(tmp_path)
    try:
        with pytest.raises(ApiError) as caught:
            server.start()
        _assert_api_code(caught, "STORAGE_IN_USE")
    finally:
        server.stop()
    assert endpoint.read_text(encoding="utf-8") == "preserve"

    unprepared = _server(tmp_path / "other")
    with pytest.raises(RuntimeError, match="not prepared"):
        unprepared._serve_unix()


@pytest.mark.skipif(os.name == "nt", reason="Unix-domain socket invariant")
def test_failed_private_mode_check_removes_the_exact_socket_it_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _server(tmp_path)
    chmod = control.os.chmod

    def leave_new_socket_unsafe(path: os.PathLike[str] | str, mode: int) -> None:
        if Path(path) != server.socket_path:
            chmod(path, mode)

    monkeypatch.setattr(control.os, "chmod", leave_new_socket_unsafe)
    with pytest.raises(ApiError) as caught:
        server.start()
    _assert_api_code(caught, "STORAGE_UNAVAILABLE")

    server.stop()
    assert not server.control_path.exists()
    assert not server.socket_path.exists()


@pytest.mark.skipif(os.name == "nt", reason="Unix-domain socket invariant")
def test_failed_private_mode_check_retains_a_replacement_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _server(tmp_path)
    chmod = control.os.chmod
    replacement: socket.socket | None = None

    def replace_new_socket(path: os.PathLike[str] | str, mode: int) -> None:
        nonlocal replacement
        target = Path(path)
        if target != server.socket_path:
            chmod(path, mode)
            return
        target.unlink()
        replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        replacement.bind(str(target))

    monkeypatch.setattr(control.os, "chmod", replace_new_socket)
    try:
        with pytest.raises(ApiError) as caught:
            server.start()
        _assert_api_code(caught, "STORAGE_UNAVAILABLE")
        server.stop()
        assert server.socket_path.exists()
    finally:
        if replacement is not None:
            replacement.close()
        server.socket_path.unlink(missing_ok=True)


def test_windows_api_configuration_and_loader_are_source_only_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _DLL()
    advapi = _DLL()
    control._configure_windows_apis(kernel, advapi)
    for name in (
        "GetCurrentProcess",
        "CloseHandle",
        "ReadFile",
        "WriteFile",
        "PeekNamedPipe",
        "WaitNamedPipeW",
        "CreateFileW",
    ):
        assert kernel._calls[name].argtypes is not None
        assert kernel._calls[name].restype is not None
    assert advapi._calls["GetTokenInformation"].argtypes is not None

    with pytest.raises(ApiError) as unavailable:
        control._windows_apis()
    _assert_api_code(unavailable, "CAPABILITY_UNAVAILABLE")

    userenv = _DLL()
    libraries = {"kernel32": kernel, "advapi32": advapi, "userenv": userenv}
    configured: list[tuple[Any, Any]] = []
    monkeypatch.setattr(control, "os", _OSProxy("nt"))
    monkeypatch.setattr(
        control.ctypes,
        "WinDLL",
        lambda name, use_last_error=True: libraries[name],
        raising=False,
    )
    monkeypatch.setattr(
        control,
        "_configure_windows_apis",
        lambda left, right: configured.append((left, right)),
    )
    assert control._windows_apis() == (kernel, advapi, userenv)
    assert configured == [(kernel, advapi)]


def test_windows_sid_and_pipe_security_descriptor_with_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # These fakes verify ctypes ownership/error paths; they are not native-Windows evidence.
    kernel = _DLL()
    advapi = _DLL()
    sid_text = "S-1-5-21-1000"

    def open_token(_process, _rights, output) -> int:
        output._obj.value = 77
        return 1

    def token_info(_token, _kind, buffer, _size, needed) -> int:
        if buffer is None:
            needed._obj.value = ctypes.sizeof(ctypes.c_void_p)
            return 0
        value = ctypes.c_void_p(0x1234)
        ctypes.memmove(buffer, ctypes.byref(value), ctypes.sizeof(value))
        return 1

    def sid_to_text(_sid, output) -> int:
        output._obj.value = sid_text
        return 1

    kernel.set("GetCurrentProcess", _Call(99))
    kernel.set("CloseHandle", _Call(1))
    kernel.set("LocalFree", _Call(0))
    advapi.set("OpenProcessToken", _Call(effect=open_token))
    advapi.set("GetTokenInformation", _Call(effect=token_info))
    advapi.set("ConvertSidToStringSidW", _Call(effect=sid_to_text))
    monkeypatch.setattr(control, "_windows_apis", lambda: (kernel, advapi, _DLL()))
    assert control._windows_current_user_sid() == sid_text
    assert kernel.CloseHandle.calls
    assert kernel.LocalFree.calls

    def convert_sddl(_sddl, _revision, output, _size) -> int:
        output._obj.value = 0x2222
        return 1

    advapi.set(
        "ConvertStringSecurityDescriptorToSecurityDescriptorW",
        _Call(effect=convert_sddl),
    )
    kernel.set("CreateNamedPipeW", _Call(123))
    monkeypatch.setattr(control, "_windows_current_user_sid", lambda: sid_text)
    assert control._create_windows_pipe("pipe", first=True) == 123
    assert kernel.CreateNamedPipeW.calls[0][1] & 0x00080000
    assert kernel.LocalFree.calls


def test_windows_pipe_io_and_connection_fail_closed_with_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _DLL()
    monkeypatch.setattr(control, "_windows_apis", lambda: (kernel, _DLL(), _DLL()))
    monkeypatch.setattr(control.ctypes, "WinError", lambda code=None: OSError(code), raising=False)

    kernel.set("ConnectNamedPipe", _Call(1))
    assert control._connect_windows_pipe(10)
    kernel.set("ConnectNamedPipe", _Call(0))
    monkeypatch.setattr(control.ctypes, "get_last_error", lambda: 535, raising=False)
    assert control._connect_windows_pipe(10)
    monkeypatch.setattr(control.ctypes, "get_last_error", lambda: 995)
    assert not control._connect_windows_pipe(10)
    monkeypatch.setattr(control.ctypes, "get_last_error", lambda: 123)
    with pytest.raises(OSError):
        control._connect_windows_pipe(10)

    chunks = [b'{"ok":true}\n']

    def read_file(_handle, buffer, _size, count, _overlap) -> int:
        data = chunks.pop(0)
        ctypes.memmove(buffer, data, len(data))
        count._obj.value = len(data)
        return 1

    def peek(_handle, _buffer, _size, _read, available, _remaining) -> int:
        available._obj.value = 0
        return 1

    kernel.set("ReadFile", _Call(effect=read_file))
    kernel.set("PeekNamedPipe", _Call(effect=peek))
    assert control._read_windows_line(10) == (b'{"ok":true}', None)

    written: list[bytes] = []

    def write_file(_handle, data, size, count, _overlap) -> int:
        amount = min(size, 3)
        written.append(bytes(data[:amount]))
        count._obj.value = amount
        return 1

    kernel.set("WriteFile", _Call(effect=write_file))
    control._write_windows(10, b"abcdef")
    assert b"".join(written) == b"abcdef"

    kernel.set("FlushFileBuffers", _Call(1))
    kernel.set("DisconnectNamedPipe", _Call(1))
    kernel.set("CloseHandle", _Call(1))
    control._close_windows_pipe(10)
    control._close_windows_handle(11)
    assert len(kernel.CloseHandle.calls) == 2


def test_windows_open_exchange_and_server_loop_with_source_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _DLL()
    kernel.set("WaitNamedPipeW", _Call(1))
    kernel.set("CreateFileW", _Call(55))
    kernel.set("CloseHandle", _Call(1))
    monkeypatch.setattr(control, "_windows_apis", lambda: (kernel, _DLL(), _DLL()))
    moments = iter((10.0, 10.0))
    monkeypatch.setattr(control.time, "monotonic", lambda: next(moments))
    assert control._open_windows_pipe("pipe", 1.0) == (kernel, 55)

    monkeypatch.setattr(control, "_open_windows_pipe", lambda _name, _timeout: (kernel, 77))
    writes: list[tuple[int, bytes]] = []
    monkeypatch.setattr(control, "_write_windows", lambda handle, data: writes.append((handle, data)))
    monkeypatch.setattr(control, "_read_windows_line", lambda _handle: (b"response", None))
    assert control._exchange_windows("pipe", b"request", 1.0) == b"response"
    assert writes == [(77, b"request")]
    assert kernel.CloseHandle.calls[-1] == (77,)

    server = _server(tmp_path, {"status": lambda _args: {"safe": True}})
    server._first_pipe_handle = 88
    closed: list[int] = []
    responses: list[bytes] = []
    monkeypatch.setattr(control, "_connect_windows_pipe", lambda _handle: True)
    monkeypatch.setattr(control, "_read_windows_line", lambda _handle: (_request("status"), None))

    def capture(handle: int, data: bytes) -> None:
        responses.append(data)
        server._stop_event.set()

    monkeypatch.setattr(control, "_write_windows", capture)
    monkeypatch.setattr(control, "_close_windows_pipe", lambda handle: closed.append(handle))
    server._serve_windows()
    assert json.loads(responses[0])["result"] == {"safe": True}
    assert closed == [88]
    assert server._pipe_handle is None


def test_windows_stop_wakes_or_closes_only_owned_handles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control, "os", _OSProxy("nt"))
    wakes: list[str] = []
    closes: list[int] = []
    monkeypatch.setattr(
        control,
        "_exchange_windows",
        lambda endpoint, _request, _timeout: wakes.append(endpoint),
    )
    monkeypatch.setattr(control, "_close_windows_handle", lambda handle: closes.append(handle))

    active = _server(tmp_path / "active")
    active._pipe_handle = 7
    active.stop()
    assert wakes == [active.descriptor.endpoint]

    prepared = _server(tmp_path / "prepared")
    prepared._first_pipe_handle = 9
    prepared.stop()
    assert closes == [9]



def test_additional_control_identity_bootstrap_and_secret_failures(tmp_path: Path) -> None:
    assert not control._canonical_instance_id("not-a-uuid")
    grant = auth.BootstrapGrant("a" * 43, 1.0, 2.0)
    assert control._safe_result(grant) == {"url": grant.url}
    with pytest.raises(ValueError, match="UUIDv4"):
        control.ControlServer(tmp_path, "not-a-uuid", os.getpid(), _SECRET, {})

    server = _server(tmp_path)
    assert _error_code(
        server.handle_line(
            canonical_json(
                {
                    "control_secret": "x" * 43,
                    "operation": "status",
                    "arguments": {},
                }
            )
        )
    ) == "AUTH_REQUIRED"
    assert _error_code(server.handle_line(_request("unknown"))) == "INVALID_REQUEST"


def test_control_context_removes_its_owned_descriptor_and_socket(tmp_path: Path) -> None:
    server = _server(
        tmp_path,
        {"status": lambda _args: {"instance_id": _INSTANCE_ID, "pid": os.getpid()}},
    )
    with server as active:
        assert active is server
        assert server.control_path.exists()
        assert server.socket_path.exists()
    assert not server.control_path.exists()
    assert not server.socket_path.exists()

    # Cleanup tolerates an endpoint already removed by an external recovery step.
    server._socket_identity = (1, 1)
    server._remove_published_files()


def test_unix_failed_prepare_handles_disappearing_owned_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _server(tmp_path)
    chmod = control.os.chmod

    def remove_endpoint(path: os.PathLike[str] | str, mode: int) -> None:
        target = Path(path)
        if target == server.socket_path:
            target.unlink()
        else:
            chmod(path, mode)

    monkeypatch.setattr(control.os, "chmod", remove_endpoint)
    with pytest.raises(FileNotFoundError):
        server.start()
    server.stop()
    assert not server.socket_path.exists()


def test_unix_server_loop_handles_timeout_send_failure_and_listener_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def settimeout(self, _value: float) -> None:
            pass

        def sendall(self, _value: bytes) -> None:
            raise OSError("peer left")

    class Listener:
        def __init__(self) -> None:
            self.actions: list[Any] = [TimeoutError(), (Connection(), None), OSError("listener")]

        def accept(self):
            value = self.actions.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value

    server = _server(tmp_path)
    actual_handle = server._handle_socket_connection
    server._listener = Listener()  # type: ignore[assignment]
    monkeypatch.setattr(server, "_handle_socket_connection", lambda _connection: b"{}\n")
    with pytest.raises(OSError, match="listener"):
        server._serve_unix()

    failure = control._protocol_error("INVALID_REQUEST", "bad frame")
    monkeypatch.setattr(control, "_read_socket_line", lambda _connection: (b"", failure))
    assert json.loads(actual_handle(object()))["error"]["code"] == "INVALID_REQUEST"  # type: ignore[arg-type]


def test_line_reader_covers_oversize_newline_and_nonblocking_peek_error() -> None:
    oversized = _ScriptedSocket([b"x" * (control.MAX_CONTROL_LINE_BYTES + 1) + b"\n"])
    line, failure = control._read_socket_line(oversized)
    assert line == b""
    assert _error_code(failure or {}) == "PAYLOAD_TOO_LARGE"

    peek_error = _ScriptedSocket([b"{}\n"], peek=BlockingIOError())  # type: ignore[arg-type]
    assert control._read_socket_line(peek_error) == (b"{}", None)


def test_control_client_public_call_pair_and_named_pipe_source_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = control.ControlClient(tmp_path)
    descriptor = control.ControlDescriptor(
        _INSTANCE_ID,
        os.getpid(),
        _SECRET,
        "unix",
        str(tmp_path / "control.sock"),
    )
    monkeypatch.setattr(client, "descriptor", lambda: descriptor)
    monkeypatch.setattr(client, "_call_with_descriptor", lambda *_args: {"ok": "public"})
    assert client.call("status", {}) == {"ok": "public"}

    calls: list[tuple[str, bytes, float]] = []
    named = control.ControlDescriptor(
        _INSTANCE_ID,
        os.getpid(),
        _SECRET,
        "named_pipe",
        rf"\\.\pipe\llm-foundations-{_INSTANCE_ID}",
    )
    monkeypatch.setattr(control, "os", _OSProxy("nt"))
    monkeypatch.setattr(
        control,
        "_exchange_windows",
        lambda endpoint, request, timeout: (
            calls.append((endpoint, request, timeout))
            or canonical_json({"ok": True, "result": 9})
        ),
    )
    named_client = control.ControlClient(tmp_path)
    assert named_client._call_with_descriptor(named, "status", {}) == 9
    assert calls and calls[0][0] == named.endpoint

    monkeypatch.setattr(
        client,
        "_call_with_descriptor",
        lambda _descriptor, operation, _arguments: (
            {"instance_id": _INSTANCE_ID, "pid": descriptor.pid}
            if operation == "status"
            else {"url": "http://127.0.0.1:8765/#/connect/value"}
        ),
    )
    assert client.pair()["url"].endswith("/value")


def test_control_client_rejects_transport_failure_and_pair_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = control.ControlClient(tmp_path)
    endpoint = tmp_path / "control.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(endpoint))
    endpoint.chmod(0o600)

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def settimeout(self, _value):
            pass

        def connect(self, _endpoint):
            pass

        def sendall(self, _request):
            pass

        def shutdown(self, _direction):
            pass

    failure = control._protocol_error("INVALID_REQUEST", "bad response frame")
    monkeypatch.setattr(control.socket, "socket", lambda *_args: FakeConnection())
    monkeypatch.setattr(control, "_read_socket_line", lambda _connection: (b"", failure))
    try:
        with pytest.raises(ApiError) as caught:
            client._exchange_unix(str(endpoint), b"{}\n")
        _assert_api_code(caught, "INVALID_REQUEST")
    finally:
        listener.close()
        endpoint.unlink(missing_ok=True)

    descriptor = control.ControlDescriptor(
        _INSTANCE_ID,
        os.getpid(),
        _SECRET,
        "unix",
        str(tmp_path / "control.sock"),
    )
    monkeypatch.setattr(client, "descriptor", lambda: descriptor)
    monkeypatch.setattr(client, "_call_with_descriptor", lambda *_args: {})
    with pytest.raises(ApiError) as mismatch:
        client.pair()
    _assert_api_code(mismatch, "AUTH_REQUIRED")


def test_invalid_control_error_code_is_never_trusted() -> None:
    malformed = {
        "ok": False,
        "error": {
            "code": "NOT_A_DECLARED_ERROR",
            "message": "bad",
            "field_errors": [],
            "retryable": False,
            "request_id": _INSTANCE_ID,
        },
    }
    with pytest.raises(ApiError) as caught:
        control._decode_control_response(canonical_json(malformed))
    _assert_api_code(caught, "INVALID_REQUEST")


def test_windows_prepare_dispatch_and_disconnected_rotation_use_source_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control, "os", _OSProxy("nt"))
    monkeypatch.setattr(control, "ensure_private_directory", lambda _path: None)
    monkeypatch.setattr(control, "write_private_file", lambda _path, _data: None)
    created: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        control,
        "_create_windows_pipe",
        lambda endpoint, first: (created.append((endpoint, first)) or 41),
    )
    server = _server(tmp_path)
    server._prepare()
    server._prepare()
    assert created == [(server.descriptor.endpoint, True)]

    dispatched: list[bool] = []
    monkeypatch.setattr(server, "_serve_windows", lambda: dispatched.append(True))
    server.serve_forever()
    assert dispatched == [True]

    rotating = _server(tmp_path / "rotating")
    rotating._prepared = True
    handles = iter((51, 52))
    closed: list[int] = []
    monkeypatch.setattr(control, "_create_windows_pipe", lambda _endpoint, first: next(handles))

    def disconnected(handle: int) -> bool:
        if handle == 52:
            rotating._stop_event.set()
        return False

    monkeypatch.setattr(control, "_connect_windows_pipe", disconnected)
    monkeypatch.setattr(control, "_close_windows_pipe", lambda handle: closed.append(handle))
    rotating._serve_windows()
    assert closed == [51, 52]


def test_windows_sid_and_pipe_creation_errors_close_owned_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _DLL()
    advapi = _DLL()
    kernel.set("GetCurrentProcess", _Call(1))
    kernel.set("CloseHandle", _Call(1))
    kernel.set("LocalFree", _Call(0))
    monkeypatch.setattr(control, "_windows_apis", lambda: (kernel, advapi, _DLL()))
    monkeypatch.setattr(control.ctypes, "get_last_error", lambda: 5, raising=False)
    monkeypatch.setattr(control.ctypes, "WinError", lambda code=None: OSError(code), raising=False)

    advapi.set("OpenProcessToken", _Call(0))
    with pytest.raises(OSError):
        control._windows_current_user_sid()

    def open_token(_process, _rights, output) -> int:
        output._obj.value = 70
        return 1

    def fail_second_token(_token, _kind, buffer, _size, needed) -> int:
        if buffer is None:
            needed._obj.value = ctypes.sizeof(ctypes.c_void_p)
            return 0
        return 0

    advapi.set("OpenProcessToken", _Call(effect=open_token))
    advapi.set("GetTokenInformation", _Call(effect=fail_second_token))
    with pytest.raises(OSError):
        control._windows_current_user_sid()
    assert kernel.CloseHandle.calls

    sid_pointer = ctypes.c_void_p(0x1234)

    def token_ok(_token, _kind, buffer, _size, needed) -> int:
        if buffer is None:
            needed._obj.value = ctypes.sizeof(ctypes.c_void_p)
            return 0
        ctypes.memmove(buffer, ctypes.byref(sid_pointer), ctypes.sizeof(sid_pointer))
        return 1

    advapi.set("GetTokenInformation", _Call(effect=token_ok))
    advapi.set("ConvertSidToStringSidW", _Call(0))
    with pytest.raises(OSError):
        control._windows_current_user_sid()

    monkeypatch.setattr(control, "_windows_current_user_sid", lambda: "S-1-5-21-1")
    advapi.set("ConvertStringSecurityDescriptorToSecurityDescriptorW", _Call(0))
    with pytest.raises(OSError):
        control._create_windows_pipe("pipe", first=False)

    def descriptor_ok(_text, _revision, output, _size) -> int:
        output._obj.value = 0x1234
        return 1

    advapi.set(
        "ConvertStringSecurityDescriptorToSecurityDescriptorW",
        _Call(effect=descriptor_ok),
    )
    kernel.set("CreateNamedPipeW", _Call(ctypes.c_void_p(-1).value))
    with pytest.raises(OSError):
        control._create_windows_pipe("pipe", first=False)
    assert kernel.LocalFree.calls


def test_windows_line_and_write_failures_are_bounded_source_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _DLL()
    monkeypatch.setattr(control, "_windows_apis", lambda: (kernel, _DLL(), _DLL()))
    monkeypatch.setattr(control.ctypes, "get_last_error", lambda: 5, raising=False)
    monkeypatch.setattr(control.ctypes, "WinError", lambda code=None: OSError(code), raising=False)

    kernel.set("ReadFile", _Call(0))
    _line, failure = control._read_windows_line(1)
    assert _error_code(failure or {}) == "INVALID_REQUEST"

    def reader(chunks: list[bytes]):
        queue = list(chunks)

        def read(_handle, buffer, _size, count, _overlap):
            data = queue.pop(0)
            ctypes.memmove(buffer, data, len(data))
            count._obj.value = len(data)
            return 1

        return read

    kernel.set("ReadFile", _Call(effect=reader([b"x" * 65536] * 17)))
    _line, failure = control._read_windows_line(1)
    assert _error_code(failure or {}) == "PAYLOAD_TOO_LARGE"

    kernel.set("ReadFile", _Call(effect=reader([b"{}\nextra"])))
    _line, failure = control._read_windows_line(1)
    assert _error_code(failure or {}) == "INVALID_REQUEST"

    kernel.set("ReadFile", _Call(effect=reader([b"{}\n"])))

    def queued(_handle, _buffer, _size, _read, available, _remaining):
        available._obj.value = 1
        return 1

    kernel.set("PeekNamedPipe", _Call(effect=queued))
    _line, failure = control._read_windows_line(1)
    assert _error_code(failure or {}) == "INVALID_REQUEST"

    kernel.set("WriteFile", _Call(0))
    with pytest.raises(OSError):
        control._write_windows(1, b"x")

    def no_progress(_handle, _data, _size, count, _overlap):
        count._obj.value = 0
        return 1

    kernel.set("WriteFile", _Call(effect=no_progress))
    with pytest.raises(OSError, match="no write progress"):
        control._write_windows(1, b"x")


def test_windows_open_and_exchange_retry_and_failure_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _DLL()
    monkeypatch.setattr(control, "_windows_apis", lambda: (kernel, _DLL(), _DLL()))
    monkeypatch.setattr(control.ctypes, "WinError", lambda code=None: OSError(code), raising=False)
    monkeypatch.setattr(control.time, "sleep", lambda _seconds: None)

    moments = iter((0.0, 2.0))
    monkeypatch.setattr(control.time, "monotonic", lambda: next(moments))
    with pytest.raises(ApiError) as timeout:
        control._open_windows_pipe("pipe", 1.0)
    _assert_api_code(timeout, "STORAGE_UNAVAILABLE")

    kernel.set("WaitNamedPipeW", _Call(0))
    moments = iter((0.0, 0.0))
    monkeypatch.setattr(control.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(control.ctypes, "get_last_error", lambda: 5, raising=False)
    with pytest.raises(ApiError) as unavailable:
        control._open_windows_pipe("pipe", 1.0)
    _assert_api_code(unavailable, "STORAGE_UNAVAILABLE")

    invalid = ctypes.c_void_p(-1).value
    kernel.set("WaitNamedPipeW", _Call(1))
    kernel.set("CreateFileW", _Call(invalid))
    moments = iter((0.0, 0.0))
    monkeypatch.setattr(control.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(control.ctypes, "get_last_error", lambda: 5, raising=False)
    with pytest.raises(ApiError) as forbidden:
        control._open_windows_pipe("pipe", 1.0)
    _assert_api_code(forbidden, "AUTH_REQUIRED")

    closer = _DLL()
    closer.set("CloseHandle", _Call(1))
    monkeypatch.setattr(control, "_open_windows_pipe", lambda _name, _timeout: (closer, 91))
    failure = control._protocol_error("INVALID_REQUEST", "bad pipe response")
    monkeypatch.setattr(control, "_write_windows", lambda _handle, _request: None)
    monkeypatch.setattr(control, "_read_windows_line", lambda _handle: (b"", failure))
    with pytest.raises(ApiError) as exchange:
        control._exchange_windows("pipe", b"request", 1.0)
    _assert_api_code(exchange, "INVALID_REQUEST")
    assert closer.CloseHandle.calls == [(91,)]
