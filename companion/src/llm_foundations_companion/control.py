"""Owner-only local IPC for fixed companion control operations.

The transport is a Unix-domain socket on WSL/Linux and a native Windows named
pipe on Windows.  There is deliberately no TCP fallback.
"""

from __future__ import annotations

import ctypes
import hmac
import inspect
import os
import re
import socket
import stat
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .auth import BootstrapGrant
from .errors import ApiError
from .platform_security import (
    StorageSecurityError,
    check_private_file,
    ensure_private_directory,
    write_private_file,
)
from .schema import canonical_json, strict_json


MAX_CONTROL_LINE_BYTES = 1024 * 1024
_MAX_UNIX_SOCKET_PATH_BYTES = 107
CONTROL_OPERATIONS = (
    "pair",
    "status",
    "submit_job",
    "cancel_job",
    "list_events",
    "register_e01_receipt",
    "register_reuse_receipt",
)
_CONTROL_OPERATION_SET = frozenset(CONTROL_OPERATIONS)
_CONTROL_FILE_KEYS = frozenset(
    {"instance_id", "pid", "control_secret", "transport", "endpoint"}
)
_SECRET_RE = re.compile(r"^[A-Za-z0-9_-]{43}$", re.ASCII)


def _canonical_instance_id(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        return False
    return parsed.version == 4 and str(parsed) == value


@dataclass(frozen=True)
class ControlDescriptor:
    instance_id: str
    pid: int
    control_secret: str = field(repr=False)
    transport: str
    endpoint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "pid": self.pid,
            "control_secret": self.control_secret,
            "transport": self.transport,
            "endpoint": self.endpoint,
        }


ControlHandler = Callable[[Mapping[str, Any]], Any]


def _request_id() -> str:
    return str(uuid.uuid4())


def _safe_result(value: Any) -> Any:
    if isinstance(value, BootstrapGrant):
        return {"url": value.url}
    serializer = getattr(value, "as_dict", None)
    if callable(serializer):
        return serializer()
    return value


def _error_response(error: ApiError) -> dict[str, Any]:
    return {"ok": False, "error": error.as_error(_request_id())}


def _protocol_error(code: str, message: str) -> dict[str, Any]:
    return _error_response(ApiError(code, message))


def _decode_request(line: bytes) -> Mapping[str, Any]:
    try:
        request = strict_json(line)
        if canonical_json(request) != line:
            raise ValueError("request is not canonical JSON")
    except (ApiError, TypeError, ValueError, UnicodeError) as exc:
        raise ApiError(
            "INVALID_REQUEST",
            "The control request is not one canonical JSON object.",
        ) from exc

    if not isinstance(request, dict) or set(request) != {
        "control_secret",
        "operation",
        "arguments",
    }:
        raise ApiError(
            "INVALID_REQUEST",
            "The control request has an invalid shape.",
        )
    if not isinstance(request["control_secret"], str):
        raise ApiError("INVALID_REQUEST", "The control secret must be a string.")
    if not isinstance(request["operation"], str):
        raise ApiError("INVALID_REQUEST", "The control operation must be a string.")
    if not isinstance(request["arguments"], dict):
        raise ApiError("INVALID_REQUEST", "Control arguments must be an object.")
    return request


def _encode_response(response: Mapping[str, Any]) -> bytes:
    try:
        payload = canonical_json(response)
    except (ApiError, TypeError, ValueError):
        payload = canonical_json(
            _protocol_error("INTERNAL_ERROR", "The control operation failed.")
        )
    if len(payload) > MAX_CONTROL_LINE_BYTES:
        payload = canonical_json(
            _protocol_error(
                "PAYLOAD_TOO_LARGE",
                "The control response exceeds the transport limit.",
            )
        )
    return payload + b"\n"


class ControlServer:
    """Serve exactly the frozen control-operation map on owner-only IPC."""

    def __init__(
        self,
        storage_root: Path | str,
        instance_id: str,
        pid: int,
        control_secret: str,
        handlers: Mapping[str, ControlHandler],
    ) -> None:
        self.storage_root = Path(storage_root).resolve()
        self.runtime_dir = self.storage_root / "runtime"
        self.control_path = self.runtime_dir / "control.json"
        self.socket_path = self.runtime_dir / "control.sock"
        self.instance_id = instance_id
        self.pid = int(pid)
        self._control_secret = control_secret
        self._handlers = dict(handlers)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._listener: socket.socket | None = None
        self._socket_identity: tuple[int, int] | None = None
        self._pipe_handle: int | None = None
        self._first_pipe_handle: int | None = None
        self._prepared = False
        self._state_lock = threading.RLock()
        if self.pid <= 0:
            raise ValueError("pid must be positive")
        if not _canonical_instance_id(instance_id):
            raise ValueError("instance_id must be a canonical lowercase UUIDv4")
        if not _SECRET_RE.fullmatch(control_secret):
            raise ValueError("control_secret must encode exactly 256 bits")

    @property
    def descriptor(self) -> ControlDescriptor:
        transport = "named_pipe" if os.name == "nt" else "unix"
        endpoint = (
            rf"\\.\pipe\llm-foundations-{self.instance_id}"
            if os.name == "nt"
            else str(self.socket_path)
        )
        return ControlDescriptor(
            instance_id=self.instance_id,
            pid=self.pid,
            control_secret=self._control_secret,
            transport=transport,
            endpoint=endpoint,
        )

    def _prepare(self) -> None:
        with self._state_lock:
            if self._prepared:
                return
            ensure_private_directory(self.runtime_dir)
            if os.name == "nt":
                self._first_pipe_handle = _create_windows_pipe(
                    self.descriptor.endpoint,
                    first=True,
                )
            else:
                self._prepare_unix()
            write_private_file(
                self.control_path,
                canonical_json(self.descriptor.as_dict()) + b"\n",
            )
            self._prepared = True

    def _prepare_unix(self) -> None:
        if len(os.fsencode(str(self.socket_path))) > _MAX_UNIX_SOCKET_PATH_BYTES:
            raise ApiError(
                "STORAGE_UNAVAILABLE",
                "The storage root is too long for local control; choose a shorter root.",
            )
        if self.socket_path.exists() or self.socket_path.is_symlink():
            existing = self.socket_path.lstat()
            if not stat.S_ISSOCK(existing.st_mode):
                raise ApiError(
                    "STORAGE_IN_USE",
                    "The private control endpoint is occupied.",
                )
            self.socket_path.unlink()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bound_identity: tuple[int, int] | None = None
        try:
            listener.bind(str(self.socket_path))
            bound_info = self.socket_path.lstat()
            bound_identity = (bound_info.st_dev, bound_info.st_ino)
            os.chmod(self.socket_path, 0o600)
            listener.listen(8)
            listener.settimeout(0.25)
            info = self.socket_path.lstat()
            if (
                (info.st_dev, info.st_ino) != bound_identity
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                raise ApiError(
                    "STORAGE_UNAVAILABLE",
                    "The private control endpoint permissions are unsafe.",
                )
            self._socket_identity = bound_identity
            self._listener = listener
        except BaseException:
            listener.close()
            if bound_identity is not None:
                try:
                    current = self.socket_path.lstat()
                    if (current.st_dev, current.st_ino) == bound_identity:
                        self.socket_path.unlink()
                except FileNotFoundError:
                    pass
            raise

    def start(self) -> ControlDescriptor:
        """Publish the private endpoint and serve it on a background thread."""

        self._prepare()
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return self.descriptor
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self.serve_forever,
                name="llmf-control",
                daemon=True,
            )
            self._thread.start()
        return self.descriptor

    def serve_forever(self) -> None:
        self._prepare()
        if os.name == "nt":
            self._serve_windows()
        else:
            self._serve_unix()

    def _serve_unix(self) -> None:
        listener = self._listener
        if listener is None:
            raise RuntimeError("Unix control listener was not prepared")
        while not self._stop_event.is_set():
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop_event.is_set():
                    break
                raise
            with connection:
                connection.settimeout(2.0)
                response = self._handle_socket_connection(connection)
                try:
                    connection.sendall(response)
                except OSError:
                    pass

    def _handle_socket_connection(self, connection: socket.socket) -> bytes:
        line, failure = _read_socket_line(connection)
        if failure is not None:
            return _encode_response(failure)
        return _encode_response(self.handle_line(line))

    def handle_line(self, line: bytes) -> Mapping[str, Any]:
        """Validate and dispatch one already-framed control request line."""

        try:
            request = _decode_request(line)
            supplied = request["control_secret"]
            if not hmac.compare_digest(supplied, self._control_secret):
                raise ApiError(
                    "AUTH_REQUIRED",
                    "The private control credential is invalid.",
                )
            operation = request["operation"]
            if operation not in _CONTROL_OPERATION_SET:
                raise ApiError(
                    "INVALID_REQUEST",
                    "The control operation is not recognized.",
                )
            handler = self._handlers.get(operation)
            if handler is None:
                raise ApiError(
                    "CAPABILITY_UNAVAILABLE",
                    "The control operation is not available in this build slice.",
                )
            result = handler(request["arguments"])
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise ApiError(
                    "CAPABILITY_UNAVAILABLE",
                    "The control operation requires a synchronous service adapter.",
                )
            return {"ok": True, "result": _safe_result(result)}
        except ApiError as exc:
            return _error_response(exc)
        except Exception:
            return _protocol_error("INTERNAL_ERROR", "The control operation failed.")

    def _serve_windows(self) -> None:
        first = self._first_pipe_handle
        self._first_pipe_handle = None
        while not self._stop_event.is_set():
            handle = first
            first = None
            if handle is None:
                handle = _create_windows_pipe(self.descriptor.endpoint, first=False)
            with self._state_lock:
                self._pipe_handle = handle
            try:
                if not _connect_windows_pipe(handle):
                    if self._stop_event.is_set():
                        break
                    continue
                line, failure = _read_windows_line(handle)
                response = _encode_response(
                    failure if failure is not None else self.handle_line(line)
                )
                _write_windows(handle, response)
            finally:
                _close_windows_pipe(handle)
                with self._state_lock:
                    if self._pipe_handle == handle:
                        self._pipe_handle = None

    def stop(self) -> None:
        """Stop serving and remove only this server's published endpoints."""

        self._stop_event.set()
        with self._state_lock:
            listener = self._listener
            self._listener = None
            active_pipe_handle = self._pipe_handle
            first_pipe_handle = self._first_pipe_handle
            self._first_pipe_handle = None
        if listener is not None:
            listener.close()
        if active_pipe_handle is not None and os.name == "nt":
            # Wake a blocking ConnectNamedPipe through the same private pipe.
            # Avoid closing a handle from another thread, which can race with
            # Windows reusing that numeric handle value.
            try:
                _exchange_windows(self.descriptor.endpoint, b"{}\n", 1.0)
            except (ApiError, OSError):
                pass
        elif first_pipe_handle is not None and os.name == "nt":
            # Preparation may have created the first instance before publishing
            # control.json or starting the server thread.  No peer can produce a
            # response in that state, so close the unconnected handle directly.
            _close_windows_handle(first_pipe_handle)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        self._remove_published_files()

    def _remove_published_files(self) -> None:
        try:
            current = _load_descriptor(self.control_path)
        except (ApiError, OSError, StorageSecurityError, ValueError):
            current = None
        if current is not None and current.instance_id == self.instance_id:
            try:
                self.control_path.unlink()
            except FileNotFoundError:
                pass
        if os.name != "nt" and self._socket_identity is not None:
            try:
                info = self.socket_path.lstat()
                if (info.st_dev, info.st_ino) == self._socket_identity:
                    self.socket_path.unlink()
            except FileNotFoundError:
                pass
        self._prepared = False

    def __enter__(self) -> "ControlServer":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def _read_socket_line(
    connection: socket.socket,
) -> tuple[bytes, Mapping[str, Any] | None]:
    received = bytearray()
    while True:
        if len(received) > MAX_CONTROL_LINE_BYTES:
            return b"", _protocol_error(
                "PAYLOAD_TOO_LARGE",
                "The control request exceeds the transport limit.",
            )
        try:
            chunk = connection.recv(min(65536, MAX_CONTROL_LINE_BYTES + 2))
        except (TimeoutError, OSError):
            return b"", _protocol_error(
                "INVALID_REQUEST",
                "The control request must contain exactly one line.",
            )
        if not chunk:
            return b"", _protocol_error(
                "INVALID_REQUEST",
                "The control request must contain exactly one line.",
            )
        received.extend(chunk)
        newline = received.find(b"\n")
        if newline < 0:
            continue
        if newline > MAX_CONTROL_LINE_BYTES:
            return b"", _protocol_error(
                "PAYLOAD_TOO_LARGE",
                "The control request exceeds the transport limit.",
            )
        if received[newline + 1 :]:
            return b"", _protocol_error(
                "INVALID_REQUEST",
                "The control request must contain exactly one line.",
            )
        # Detect a second line already queued by the peer without logging it.
        try:
            connection.setblocking(False)
            extra = connection.recv(1, socket.MSG_PEEK)
        except (BlockingIOError, OSError):
            extra = b""
        finally:
            connection.setblocking(True)
            connection.settimeout(2.0)
        if extra:
            return b"", _protocol_error(
                "INVALID_REQUEST",
                "The control request must contain exactly one line.",
            )
        return bytes(received[:newline]), None


class ControlClient:
    """Read owner-only discovery data and issue fixed local control requests."""

    def __init__(self, storage_root: Path | str, *, timeout: float = 5.0) -> None:
        self.storage_root = Path(storage_root).resolve()
        self.control_path = self.storage_root / "runtime" / "control.json"
        self.timeout = timeout

    def descriptor(self) -> ControlDescriptor:
        return _load_descriptor(self.control_path)

    def call(self, operation: str, arguments: Mapping[str, Any]) -> Any:
        descriptor = self.descriptor()
        return self._call_with_descriptor(descriptor, operation, arguments)

    def _call_with_descriptor(
        self,
        descriptor: ControlDescriptor,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> Any:
        if operation not in _CONTROL_OPERATION_SET:
            raise ApiError("INVALID_REQUEST", "The control operation is not recognized.")
        request = {
            "control_secret": descriptor.control_secret,
            "operation": operation,
            "arguments": dict(arguments),
        }
        line = canonical_json(request)
        if len(line) > MAX_CONTROL_LINE_BYTES:
            raise ApiError(
                "PAYLOAD_TOO_LARGE",
                "The control request exceeds the transport limit.",
            )
        if descriptor.transport == "unix":
            response_line = self._exchange_unix(descriptor.endpoint, line + b"\n")
        elif descriptor.transport == "named_pipe" and os.name == "nt":
            response_line = _exchange_windows(
                descriptor.endpoint,
                line + b"\n",
                self.timeout,
            )
        else:
            raise ApiError(
                "CAPABILITY_UNAVAILABLE",
                "The recorded private control transport is unavailable here.",
            )
        return _decode_control_response(response_line)

    def _exchange_unix(self, endpoint: str, request: bytes) -> bytes:
        target = Path(endpoint)
        info = target.lstat()
        if (
            not stat.S_ISSOCK(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
        ):
            raise ApiError(
                "AUTH_REQUIRED",
                "The private control endpoint permissions are invalid.",
            )
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout)
            connection.connect(endpoint)
            connection.sendall(request)
            connection.shutdown(socket.SHUT_WR)
            line, failure = _read_socket_line(connection)
        if failure is not None:
            error = failure["error"]
            raise ApiError(error["code"], error["message"])
        return line

    def verify_instance(self) -> Mapping[str, Any]:
        descriptor = self.descriptor()
        result = self._call_with_descriptor(descriptor, "status", {})
        if not isinstance(result, dict):
            raise ApiError("AUTH_REQUIRED", "The control status response is invalid.")
        if result.get("instance_id") != descriptor.instance_id or result.get("pid") != descriptor.pid:
            raise ApiError(
                "AUTH_REQUIRED",
                "The control endpoint does not match its discovery record.",
            )
        return result

    def pair(self) -> Mapping[str, Any]:
        descriptor = self.descriptor()
        status = self._call_with_descriptor(descriptor, "status", {})
        if not isinstance(status, dict) or status.get("instance_id") != descriptor.instance_id or status.get("pid") != descriptor.pid:
            raise ApiError(
                "AUTH_REQUIRED",
                "The control endpoint does not match its discovery record.",
            )
        result = self._call_with_descriptor(descriptor, "pair", {})
        if not isinstance(result, dict) or not isinstance(result.get("url"), str):
            raise ApiError("INTERNAL_ERROR", "The pairing response is invalid.")
        return result


def _load_descriptor(path: Path) -> ControlDescriptor:
    check_private_file(path)
    raw = path.read_bytes()
    if len(raw) > MAX_CONTROL_LINE_BYTES:
        raise ApiError("PAYLOAD_TOO_LARGE", "The control discovery file is too large.")
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ApiError("INVALID_REQUEST", "The control discovery file is invalid.")
    try:
        value = strict_json(raw[:-1])
    except ApiError as exc:
        raise ApiError("INVALID_REQUEST", "The control discovery file is invalid.") from exc
    if not isinstance(value, dict) or set(value) != _CONTROL_FILE_KEYS:
        raise ApiError("INVALID_REQUEST", "The control discovery file is invalid.")
    if canonical_json(value) + b"\n" != raw:
        raise ApiError("INVALID_REQUEST", "The control discovery file is invalid.")
    instance_id = value["instance_id"]
    pid = value["pid"]
    secret = value["control_secret"]
    transport = value["transport"]
    endpoint = value["endpoint"]
    if (
        not _canonical_instance_id(instance_id)
        or not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
        or not isinstance(secret, str)
        or not _SECRET_RE.fullmatch(secret)
        or not isinstance(transport, str)
        or not isinstance(endpoint, str)
        or transport not in {"unix", "named_pipe"}
    ):
        raise ApiError("INVALID_REQUEST", "The control discovery file is invalid.")
    if transport == "unix" and endpoint != str(path.parent / "control.sock"):
        raise ApiError("INVALID_REQUEST", "The control discovery file is invalid.")
    expected_pipe = rf"\\.\pipe\llm-foundations-{instance_id}"
    if transport == "named_pipe" and endpoint != expected_pipe:
        raise ApiError("INVALID_REQUEST", "The control discovery file is invalid.")
    return ControlDescriptor(instance_id, pid, secret, transport, endpoint)


def _decode_control_response(line: bytes) -> Any:
    try:
        value = strict_json(line)
    except ApiError as exc:
        raise ApiError("INVALID_REQUEST", "The control response is invalid.") from exc
    if canonical_json(value) != line or not isinstance(value, dict):
        raise ApiError("INVALID_REQUEST", "The control response is invalid.")
    if value.get("ok") is True and set(value) == {"ok", "result"}:
        return value["result"]
    if value.get("ok") is False and set(value) == {"ok", "error"}:
        error = value["error"]
        required = {
            "code",
            "message",
            "field_errors",
            "retryable",
            "request_id",
        }
        if not isinstance(error, dict) or set(error) not in {
            frozenset(required),
            frozenset(required | {"reason_code"}),
        }:
            raise ApiError("INVALID_REQUEST", "The control response is invalid.")
        try:
            decoded = ApiError(
                error["code"],
                error["message"],
                reason_code=error.get("reason_code"),
                field_errors=error["field_errors"],
            )
            expected = decoded.as_error(error["request_id"])
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ApiError(
                "INVALID_REQUEST", "The control response is invalid."
            ) from exc
        if expected != error:
            raise ApiError("INVALID_REQUEST", "The control response is invalid.")
        raise decoded
    raise ApiError("INVALID_REQUEST", "The control response is invalid.")


# Windows named-pipe support is implemented with the standard-library ctypes
# boundary so installation does not gain an undeclared pywin32 dependency.
def _windows_apis() -> tuple[Any, Any, Any]:
    if os.name != "nt":
        raise ApiError(
            "CAPABILITY_UNAVAILABLE",
            "Windows named pipes are unavailable on this platform.",
        )
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    _configure_windows_apis(kernel32, advapi32)
    return kernel32, advapi32, ctypes.WinDLL("userenv", use_last_error=True)


def _configure_windows_apis(kernel32: Any, advapi32: Any) -> None:
    from ctypes import wintypes

    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL
    kernel32.ReadFile.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    kernel32.ReadFile.restype = wintypes.BOOL
    kernel32.WriteFile.argtypes = [
        wintypes.HANDLE,
        wintypes.LPCVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    kernel32.WriteFile.restype = wintypes.BOOL
    kernel32.PeekNamedPipe.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.PeekNamedPipe.restype = wintypes.BOOL
    kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
    kernel32.FlushFileBuffers.restype = wintypes.BOOL
    kernel32.DisconnectNamedPipe.argtypes = [wintypes.HANDLE]
    kernel32.DisconnectNamedPipe.restype = wintypes.BOOL
    kernel32.CancelIoEx.argtypes = [wintypes.HANDLE, wintypes.LPVOID]
    kernel32.CancelIoEx.restype = wintypes.BOOL
    kernel32.WaitNamedPipeW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD]
    kernel32.WaitNamedPipeW.restype = wintypes.BOOL
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL


def _windows_current_user_sid() -> str:
    from ctypes import wintypes

    kernel32, advapi32, _ = _windows_apis()

    class SID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Sid", wintypes.LPVOID), ("Attributes", wintypes.DWORD)]

    class TOKEN_USER(ctypes.Structure):
        _fields_ = [("User", SID_AND_ATTRIBUTES)]

    token = wintypes.HANDLE()
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    if not advapi32.OpenProcessToken(
        kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        needed = wintypes.DWORD()
        advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(needed))
        buffer = ctypes.create_string_buffer(needed.value)
        if not advapi32.GetTokenInformation(
            token, 1, buffer, needed, ctypes.byref(needed)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        token_user = ctypes.cast(buffer, ctypes.POINTER(TOKEN_USER)).contents
        sid_text = wintypes.LPWSTR()
        advapi32.ConvertSidToStringSidW.argtypes = [
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.LPWSTR),
        ]
        advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
        if not advapi32.ConvertSidToStringSidW(
            token_user.User.Sid, ctypes.byref(sid_text)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return sid_text.value
        finally:
            kernel32.LocalFree(sid_text)
    finally:
        kernel32.CloseHandle(token)


def _create_windows_pipe(name: str, *, first: bool) -> int:
    from ctypes import wintypes

    kernel32, advapi32, _ = _windows_apis()

    class SECURITY_ATTRIBUTES(ctypes.Structure):
        _fields_ = [
            ("nLength", wintypes.DWORD),
            ("lpSecurityDescriptor", wintypes.LPVOID),
            ("bInheritHandle", wintypes.BOOL),
        ]

    sid = _windows_current_user_sid()
    sddl = f"O:{sid}D:P(A;;GA;;;{sid})(A;;GA;;;SY)(A;;GA;;;BA)"
    security_descriptor = wintypes.LPVOID()
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
    if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        sddl, 1, ctypes.byref(security_descriptor), None
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    attributes = SECURITY_ATTRIBUTES(
        ctypes.sizeof(SECURITY_ATTRIBUTES), security_descriptor, False
    )
    kernel32.CreateNamedPipeW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(SECURITY_ATTRIBUTES),
    ]
    kernel32.CreateNamedPipeW.restype = wintypes.HANDLE
    open_mode = 0x00000003 | (0x00080000 if first else 0)
    try:
        handle = kernel32.CreateNamedPipeW(
            name,
            open_mode,
            0x00000008,  # PIPE_REJECT_REMOTE_CLIENTS
            1,
            65536,
            65536,
            5000,
            ctypes.byref(attributes),
        )
    finally:
        kernel32.LocalFree(security_descriptor)
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(handle)


def _connect_windows_pipe(handle: int) -> bool:
    from ctypes import wintypes

    kernel32, _, _ = _windows_apis()
    kernel32.ConnectNamedPipe.argtypes = [wintypes.HANDLE, wintypes.LPVOID]
    kernel32.ConnectNamedPipe.restype = wintypes.BOOL
    if kernel32.ConnectNamedPipe(handle, None):
        return True
    error = ctypes.get_last_error()
    if error == 535:  # ERROR_PIPE_CONNECTED
        return True
    if error in {6, 995}:  # invalid handle / operation aborted during stop
        return False
    raise ctypes.WinError(error)


def _read_windows_line(handle: int) -> tuple[bytes, Mapping[str, Any] | None]:
    from ctypes import wintypes

    kernel32, _, _ = _windows_apis()
    received = bytearray()
    buffer = ctypes.create_string_buffer(65536)
    count = wintypes.DWORD()
    while True:
        if len(received) > MAX_CONTROL_LINE_BYTES:
            return b"", _protocol_error(
                "PAYLOAD_TOO_LARGE", "The control request exceeds the transport limit."
            )
        if not kernel32.ReadFile(handle, buffer, len(buffer), ctypes.byref(count), None):
            return b"", _protocol_error(
                "INVALID_REQUEST", "The control request must contain exactly one line."
            )
        received.extend(buffer.raw[: count.value])
        newline = received.find(b"\n")
        if newline < 0:
            continue
        if newline > MAX_CONTROL_LINE_BYTES:
            return b"", _protocol_error(
                "PAYLOAD_TOO_LARGE", "The control request exceeds the transport limit."
            )
        if received[newline + 1 :]:
            return b"", _protocol_error(
                "INVALID_REQUEST", "The control request must contain exactly one line."
            )
        available = wintypes.DWORD()
        if kernel32.PeekNamedPipe(
            handle, None, 0, None, ctypes.byref(available), None
        ) and available.value:
            return b"", _protocol_error(
                "INVALID_REQUEST", "The control request must contain exactly one line."
            )
        return bytes(received[:newline]), None


def _write_windows(handle: int, data: bytes) -> None:
    from ctypes import wintypes

    kernel32, _, _ = _windows_apis()
    offset = 0
    while offset < len(data):
        count = wintypes.DWORD()
        chunk = data[offset : offset + 65536]
        if not kernel32.WriteFile(handle, chunk, len(chunk), ctypes.byref(count), None):
            raise ctypes.WinError(ctypes.get_last_error())
        if count.value == 0:
            raise OSError("the private control pipe made no write progress")
        offset += count.value


def _close_windows_pipe(handle: int) -> None:
    kernel32, _, _ = _windows_apis()
    kernel32.FlushFileBuffers(handle)
    kernel32.DisconnectNamedPipe(handle)
    kernel32.CloseHandle(handle)


def _close_windows_handle(handle: int) -> None:
    kernel32, _, _ = _windows_apis()
    kernel32.CloseHandle(handle)


def _open_windows_pipe(name: str, timeout: float) -> tuple[Any, int]:
    kernel32, _, _ = _windows_apis()
    invalid = ctypes.c_void_p(-1).value
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ApiError(
                "STORAGE_UNAVAILABLE", "The private control pipe is unavailable."
            )
        wait_ms = max(1, min(int(remaining * 1000), 60_000))
        if not kernel32.WaitNamedPipeW(name, wait_ms):
            error = ctypes.get_last_error()
            if error == 2:  # ERROR_FILE_NOT_FOUND between server pipe instances
                time.sleep(min(0.01, remaining))
                continue
            raise ApiError(
                "STORAGE_UNAVAILABLE", "The private control pipe is unavailable."
            )
        handle = kernel32.CreateFileW(
            name,
            0xC0000000,
            0,
            None,
            3,
            0,
            None,
        )
        if handle != invalid:
            return kernel32, int(handle)
        error = ctypes.get_last_error()
        if error in {2, 231}:  # server rotated the sole instance after the wait
            time.sleep(min(0.01, remaining))
            continue
        raise ApiError(
            "AUTH_REQUIRED", "The private control pipe could not be opened."
        )


def _exchange_windows(name: str, request: bytes, timeout: float) -> bytes:
    kernel32, handle = _open_windows_pipe(name, timeout)
    try:
        _write_windows(handle, request)
        line, failure = _read_windows_line(handle)
        if failure is not None:
            error = failure["error"]
            raise ApiError(error["code"], error["message"])
        return line
    finally:
        kernel32.CloseHandle(handle)


__all__ = [
    "CONTROL_OPERATIONS",
    "MAX_CONTROL_LINE_BYTES",
    "ControlClient",
    "ControlDescriptor",
    "ControlServer",
]
