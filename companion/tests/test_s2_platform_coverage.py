from __future__ import annotations

import ctypes
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO

import pytest

import llm_foundations_companion.platform_security as security


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


class _PartialWriter:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.data = bytearray()

    def write(self, value: memoryview) -> int:
        count = min(self.limit, len(value))
        self.data.extend(value[:count])
        return count


class _NoProgressWriter:
    def write(self, _value: memoryview) -> int:
        return 0


def _assert_storage_error(caught: pytest.ExceptionInfo[security.StorageSecurityError], code: str) -> None:
    assert caught.value.code == code


def test_posix_root_and_private_paths_reject_links_types_and_broad_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(security.StorageSecurityError) as absent:
        security.resolve_storage_root(missing)
    _assert_storage_error(absent, "STORAGE_UNAVAILABLE")

    root = security.resolve_storage_root(missing, create=True)
    assert root == missing.resolve()
    assert stat.S_IMODE(root.stat().st_mode) == 0o700

    regular = tmp_path / "regular"
    regular.write_text("data", encoding="utf-8")
    with pytest.raises(security.StorageSecurityError) as nondirectory:
        security.resolve_storage_root(regular)
    _assert_storage_error(nondirectory, "STORAGE_UNAVAILABLE")

    link = tmp_path / "linked-root"
    link.symlink_to(root, target_is_directory=True)
    assert security.is_reparse_point(link)
    assert not security.is_reparse_point(tmp_path / "does-not-exist")
    with pytest.raises(security.StorageSecurityError) as linked:
        security.resolve_storage_root(link)
    _assert_storage_error(linked, "STORAGE_UNAVAILABLE")

    root.chmod(0o707)
    with pytest.raises(security.StorageSecurityError) as broad:
        security.resolve_storage_root(root)
    _assert_storage_error(broad, "STORAGE_UNAVAILABLE")
    root.chmod(0o700)

    private = root / "one" / "two"
    security.ensure_private_directory(private)
    assert stat.S_IMODE(private.stat().st_mode) == 0o700

    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = root / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)
    with pytest.raises(security.StorageSecurityError) as parent_link:
        security.ensure_private_directory(linked_parent / "child")
    _assert_storage_error(parent_link, "STORAGE_UNAVAILABLE")

    not_directory = root / "not-directory"
    not_directory.write_bytes(b"file")
    with pytest.raises((FileExistsError, security.StorageSecurityError)):
        security.ensure_private_directory(not_directory)

    real_lstat = security.os.lstat
    owned = real_lstat(root)
    monkeypatch.setattr(
        security.os,
        "lstat",
        lambda _path: SimpleNamespace(st_uid=owned.st_uid + 1, st_mode=owned.st_mode),
    )
    with pytest.raises(security.StorageSecurityError) as wrong_owner:
        security._check_posix(root, private=False)
    _assert_storage_error(wrong_owner, "STORAGE_UNAVAILABLE")


def test_private_file_write_is_atomic_and_failure_preserves_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = security.resolve_storage_root(tmp_path / "root", create=True)
    target = root / "private" / "value.bin"
    security.write_private_file(target, b"first")
    assert target.read_bytes() == b"first"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    security.check_private_file(target)

    target.chmod(0o640)
    with pytest.raises(security.StorageSecurityError) as broad:
        security.check_private_file(target)
    _assert_storage_error(broad, "STORAGE_UNAVAILABLE")
    target.chmod(0o600)

    destination = root / "outside"
    destination.write_bytes(b"preserve")
    linked = root / "private" / "linked"
    linked.symlink_to(destination)
    with pytest.raises(security.StorageSecurityError):
        security.write_private_file(linked, b"replace")
    assert destination.read_bytes() == b"preserve"

    def fail_write(_stream: BinaryIO, _data: bytes) -> None:
        raise OSError("forced write failure")

    monkeypatch.setattr(security, "_write_all", fail_write)
    with pytest.raises(OSError, match="forced write"):
        security.write_private_file(target, b"second")
    assert target.read_bytes() == b"first"
    assert not tuple(target.parent.glob(".private-*"))


def test_write_all_handles_partial_progress_and_rejects_zero_progress() -> None:
    partial = _PartialWriter(2)
    security._write_all(partial, b"abcdef")  # type: ignore[arg-type]
    assert partial.data == b"abcdef"
    with pytest.raises(OSError, match="short write"):
        security._write_all(_NoProgressWriter(), b"x")  # type: ignore[arg-type]


def test_instance_lease_is_exclusive_inheritable_only_on_request_and_reusable(
    tmp_path: Path,
) -> None:
    root = security.resolve_storage_root(tmp_path / "root", create=True)
    first = security.InstanceLease(root)
    with pytest.raises(RuntimeError, match="not acquired"):
        _ = first.fileno

    assert first.acquire() is first
    descriptor = first.fileno
    assert first.acquire() is first
    assert not os.get_inheritable(descriptor)
    assert first.make_inheritable() == descriptor
    assert os.get_inheritable(descriptor)
    assert first.path.read_text(encoding="ascii") == f"{os.getpid()}\n"

    second = security.InstanceLease(root)
    with pytest.raises(security.StorageSecurityError) as occupied:
        second.acquire()
    _assert_storage_error(occupied, "STORAGE_IN_USE")

    first.release()
    first.release()
    with second:
        assert second.fileno >= 0
    with pytest.raises(RuntimeError, match="not acquired"):
        _ = second.fileno


def test_instance_lease_rejects_descriptor_path_identity_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = security.resolve_storage_root(tmp_path / "root", create=True)
    lease = security.InstanceLease(root)
    real_fstat = security.os.fstat

    def changed(descriptor: int):
        info = real_fstat(descriptor)
        return SimpleNamespace(
            st_mode=info.st_mode,
            st_dev=info.st_dev,
            st_ino=info.st_ino + 1,
        )

    monkeypatch.setattr(security.os, "fstat", changed)
    with pytest.raises(security.StorageSecurityError) as caught:
        lease.acquire()
    _assert_storage_error(caught, "STORAGE_UNAVAILABLE")


def test_instance_lease_maps_lock_contention_without_losing_the_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fcntl

    root = security.resolve_storage_root(tmp_path / "root", create=True)

    def blocked(_descriptor: int, _operation: int) -> None:
        raise BlockingIOError("held")

    monkeypatch.setattr(fcntl, "flock", blocked)
    with pytest.raises(security.StorageSecurityError) as caught:
        security.InstanceLease(root).acquire()
    _assert_storage_error(caught, "STORAGE_IN_USE")
    assert isinstance(caught.value.__cause__, BlockingIOError)


def test_windows_security_api_signatures_are_pointer_width_safe_with_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This checks source-side declarations only; it is not native-Windows evidence.
    advapi = _DLL()
    kernel = _DLL()
    monkeypatch.setattr(security, "os", _OSProxy("nt"))
    monkeypatch.setattr(
        security.ctypes,
        "windll",
        SimpleNamespace(advapi32=advapi, kernel32=kernel),
        raising=False,
    )
    security._configure_windows_security_apis()
    for name in (
        "OpenProcessToken",
        "GetTokenInformation",
        "ConvertSidToStringSidW",
        "GetNamedSecurityInfoW",
        "GetAclInformation",
        "GetAce",
    ):
        assert advapi._calls[name].argtypes is not None
        assert advapi._calls[name].restype is not None
    for name in ("GetCurrentProcess", "CloseHandle", "LocalFree"):
        assert kernel._calls[name].argtypes is not None
        assert kernel._calls[name].restype is not None


def test_windows_sid_lifecycle_closes_token_and_frees_text_with_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    advapi = _DLL()
    kernel = _DLL()
    sid_text = "S-1-5-21-1000"

    def open_token(_process, _rights, token) -> int:
        token._obj.value = 71
        return 1

    def token_info(_token, _kind, buffer, _size, needed) -> int:
        if buffer is None:
            needed._obj.value = ctypes.sizeof(ctypes.c_void_p)
            return 0
        pointer = ctypes.c_void_p(0x1234)
        ctypes.memmove(buffer, ctypes.byref(pointer), ctypes.sizeof(pointer))
        return 1

    def sid_to_text(_sid, output) -> int:
        output._obj.value = sid_text
        return 1

    advapi.set("OpenProcessToken", _Call(effect=open_token))
    advapi.set("GetTokenInformation", _Call(effect=token_info))
    advapi.set("ConvertSidToStringSidW", _Call(effect=sid_to_text))
    kernel.set("GetCurrentProcess", _Call(99))
    kernel.set("CloseHandle", _Call(1))
    kernel.set("LocalFree", _Call(0))
    monkeypatch.setattr(
        security.ctypes,
        "windll",
        SimpleNamespace(advapi32=advapi, kernel32=kernel),
        raising=False,
    )
    monkeypatch.setattr(security, "_configure_windows_security_apis", lambda: None)
    assert security._current_windows_sid() == sid_text
    assert kernel.CloseHandle.calls[-1][0].value == 71
    assert kernel.LocalFree.calls


def test_windows_acl_enumeration_detects_null_and_foreign_writers_with_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    advapi = _DLL()
    kernel = _DLL()
    monkeypatch.setattr(
        security.ctypes,
        "windll",
        SimpleNamespace(advapi32=advapi, kernel32=kernel),
        raising=False,
    )
    monkeypatch.setattr(security, "_configure_windows_security_apis", lambda: None)
    kernel.set("LocalFree", _Call(0))

    def null_dacl(_path, _kind, _info, _owner, _group, dacl, _sacl, descriptor) -> int:
        dacl._obj.value = None
        descriptor._obj.value = 0x2000
        return 0

    advapi.set("GetNamedSecurityInfoW", _Call(effect=null_dacl))
    assert security._windows_writable_sids(tmp_path) == {"*NULL_DACL*"}
    assert kernel.LocalFree.calls

    ace_buffer = ctypes.create_string_buffer(16)
    ace_buffer[0] = 0
    ace_buffer[4:8] = (0x2).to_bytes(4, "little")

    def one_dacl(_path, _kind, _info, _owner, _group, dacl, _sacl, descriptor) -> int:
        dacl._obj.value = 0x1000
        descriptor._obj.value = 0x2000
        return 0

    def acl_info(_dacl, output, _size, _kind) -> int:
        output._obj.AceCount = 1
        return 1

    def get_ace(_dacl, _index, output) -> int:
        output._obj.value = ctypes.addressof(ace_buffer)
        return 1

    advapi.set("GetNamedSecurityInfoW", _Call(effect=one_dacl))
    advapi.set("GetAclInformation", _Call(effect=acl_info))
    advapi.set("GetAce", _Call(effect=get_ace))
    monkeypatch.setattr(security, "_sid_text", lambda pointer: f"SID-{pointer:x}")
    assert security._windows_writable_sids(tmp_path) == {
        f"SID-{ctypes.addressof(ace_buffer) + 8:x}"
    }


def test_windows_acl_application_and_validation_fail_closed_with_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(security, "_current_windows_sid", lambda: "S-1-5-21-1000")
    calls: list[list[str]] = []

    def successful_run(arguments, **_kwargs):
        calls.append(arguments)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(security.subprocess, "run", successful_run)
    security._set_windows_private_acl(root)
    assert calls[0][2] == "/inheritance:r"
    assert calls[0][4].endswith(":(OI)(CI)F")

    monkeypatch.setattr(
        security.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=5),
    )
    with pytest.raises(security.StorageSecurityError) as apply_failed:
        security._set_windows_private_acl(root)
    _assert_storage_error(apply_failed, "STORAGE_UNAVAILABLE")

    monkeypatch.setattr(
        security,
        "_windows_writable_sids",
        lambda _path: {"S-1-5-21-1000", "S-1-5-18", "S-1-5-32-544"},
    )
    security._check_windows_private_acl(root)
    monkeypatch.setattr(
        security,
        "_windows_writable_sids",
        lambda _path: {"S-1-5-21-1000", "S-1-1-0"},
    )
    with pytest.raises(security.StorageSecurityError) as foreign:
        security._check_windows_private_acl(root)
    _assert_storage_error(foreign, "STORAGE_UNAVAILABLE")

    def inspection_failed(_path: Path) -> set[str]:
        raise OSError("cannot inspect")

    monkeypatch.setattr(security, "_windows_writable_sids", inspection_failed)
    with pytest.raises(security.StorageSecurityError) as inspection:
        security._check_windows_private_acl(root)
    _assert_storage_error(inspection, "STORAGE_UNAVAILABLE")
    assert isinstance(inspection.value.__cause__, OSError)


def test_windows_public_storage_paths_apply_and_recheck_acl_with_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # A module-local OS proxy reaches Windows decisions without changing pathlib or
    # claiming that any native ACL call ran on this Linux host.
    monkeypatch.setattr(security, "os", _OSProxy("nt"))
    applied: list[Path] = []
    checked: list[Path] = []
    monkeypatch.setattr(security, "_set_windows_private_acl", lambda path: applied.append(Path(path)))
    monkeypatch.setattr(security, "_check_windows_private_acl", lambda path: checked.append(Path(path)))

    existing = tmp_path / "existing"
    existing.mkdir()
    assert security.resolve_storage_root(existing) == existing.resolve()
    assert checked[-1] == existing.resolve()

    created = tmp_path / "created"
    assert security.resolve_storage_root(created, create=True) == created.resolve()
    assert created in applied
    assert created.resolve() in checked

    private = created / "private"
    security.ensure_private_directory(private)
    assert private.absolute() in applied
    assert private.absolute() in checked

    target = private / "value.bin"
    security.write_private_file(target, b"payload")
    assert target.read_bytes() == b"payload"
    assert target.absolute() in applied
    assert target.absolute() in checked
    security.check_private_file(target)


def test_windows_instance_lease_uses_nonblocking_lock_and_releases_it_with_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    locks: list[tuple[int, int, int]] = []
    fake_msvcrt = SimpleNamespace(
        LK_NBLCK=1,
        LK_UNLCK=2,
        locking=lambda descriptor, mode, count: locks.append((descriptor, mode, count)),
    )
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(security, "os", _OSProxy("nt"))
    monkeypatch.setattr(security, "_set_windows_private_acl", lambda _path: None)
    monkeypatch.setattr(security, "_check_windows_private_acl", lambda _path: None)

    lease = security.InstanceLease(root)
    lease.acquire()
    assert locks[-1][1:] == (fake_msvcrt.LK_NBLCK, 1)
    lease.release()
    assert locks[-1][1:] == (fake_msvcrt.LK_UNLCK, 1)
