"""Storage-root confinement, private paths, and the exclusive instance lease."""

from __future__ import annotations

import ctypes
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO


class StorageSecurityError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _configure_windows_security_apis() -> None:
    """Declare every pointer-width-sensitive Win32 signature we call."""

    if os.name != "nt":
        return
    from ctypes import wintypes

    advapi32 = ctypes.windll.advapi32
    kernel32 = ctypes.windll.kernel32
    advapi32.OpenProcessToken.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    )
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    )
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.ConvertSidToStringSidW.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.LPWSTR),
    )
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    advapi32.GetNamedSecurityInfoW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    )
    advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi32.GetAclInformation.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_int,
    )
    advapi32.GetAclInformation.restype = wintypes.BOOL
    advapi32.GetAce.argtypes = (
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
    )
    advapi32.GetAce.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.argtypes = ()
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = (ctypes.c_void_p,)
    kernel32.LocalFree.restype = ctypes.c_void_p


def _reject_linked_components(path: Path, *, include_leaf: bool = True) -> None:
    """Reject an existing symlink/reparse point anywhere below the path anchor."""

    target = Path(path).absolute()
    candidates = list(reversed(target.parents))
    if include_leaf:
        candidates.append(target)
    for candidate in candidates:
        if os.path.lexists(candidate) and is_reparse_point(candidate):
            raise StorageSecurityError(
                "STORAGE_UNAVAILABLE", "storage path contains a link"
            )


def is_reparse_point(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _current_windows_sid() -> str:
    from ctypes import wintypes

    _configure_windows_security_apis()
    token = wintypes.HANDLE()
    if not ctypes.windll.advapi32.OpenProcessToken(
        ctypes.windll.kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)
    ):
        raise ctypes.WinError()
    try:
        needed = wintypes.DWORD()
        ctypes.windll.advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(needed))
        buffer = ctypes.create_string_buffer(needed.value)
        if not ctypes.windll.advapi32.GetTokenInformation(
            token, 1, buffer, needed, ctypes.byref(needed)
        ):
            raise ctypes.WinError()
        class SidAndAttributes(ctypes.Structure):
            _fields_ = [("sid", ctypes.c_void_p), ("attributes", wintypes.DWORD)]

        class TokenUser(ctypes.Structure):
            _fields_ = [("user", SidAndAttributes)]

        sid = ctypes.cast(buffer, ctypes.POINTER(TokenUser)).contents.user.sid
        return _sid_text(sid)
    finally:
        ctypes.windll.kernel32.CloseHandle(token)


def _sid_text(pointer: int | ctypes.c_void_p) -> str:
    from ctypes import wintypes

    _configure_windows_security_apis()
    text = wintypes.LPWSTR()
    sid = pointer if isinstance(pointer, ctypes.c_void_p) else ctypes.c_void_p(pointer)
    if not ctypes.windll.advapi32.ConvertSidToStringSidW(sid, ctypes.byref(text)):
        raise ctypes.WinError()
    try:
        return text.value
    finally:
        ctypes.windll.kernel32.LocalFree(text)


def _windows_writable_sids(path: Path) -> set[str]:
    """Return allow-ACE SIDs with mutation rights; a null DACL is unsafe."""

    from ctypes import wintypes

    _configure_windows_security_apis()
    dacl = ctypes.c_void_p()
    security_descriptor = ctypes.c_void_p()
    result = ctypes.windll.advapi32.GetNamedSecurityInfoW(
        str(path), 1, 0x00000004, None, None, ctypes.byref(dacl), None,
        ctypes.byref(security_descriptor),
    )
    if result:
        raise ctypes.WinError(result)
    try:
        if not dacl.value:
            return {"*NULL_DACL*"}

        class AclSize(ctypes.Structure):
            _fields_ = [
                ("AceCount", wintypes.DWORD),
                ("AclBytesInUse", wintypes.DWORD),
                ("AclBytesFree", wintypes.DWORD),
            ]

        info = AclSize()
        if not ctypes.windll.advapi32.GetAclInformation(
            dacl, ctypes.byref(info), ctypes.sizeof(info), 2
        ):
            raise ctypes.WinError()
        write_mask = (
            0x2 | 0x4 | 0x10 | 0x40 | 0x100 | 0x10000 | 0x40000
            | 0x80000 | 0x10000000 | 0x40000000
        )
        writers: set[str] = set()
        for index in range(info.AceCount):
            ace = ctypes.c_void_p()
            if not ctypes.windll.advapi32.GetAce(dacl, index, ctypes.byref(ace)):
                raise ctypes.WinError()
            header = ctypes.string_at(ace, 8)
            if header[0] == 0 and int.from_bytes(header[4:8], "little") & write_mask:
                writers.add(_sid_text(ace.value + 8))
        return writers
    finally:
        if security_descriptor.value:
            ctypes.windll.kernel32.LocalFree(security_descriptor)


def _set_windows_private_acl(path: Path) -> None:
    sid = _current_windows_sid()
    grant = f"*{sid}:(OI)(CI)F" if path.is_dir() else f"*{sid}:F"
    result = subprocess.run(
        [
            "icacls.exe",
            str(path),
            "/inheritance:r",
            "/grant:r",
            grant,
            "/remove:g",
            "*S-1-3-4",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        raise StorageSecurityError(
            "STORAGE_UNAVAILABLE", "could not apply the current-user storage ACL"
        )


def _check_windows_private_acl(path: Path) -> None:
    allowed = {_current_windows_sid(), "S-1-5-18", "S-1-5-32-544"}
    try:
        unexpected = _windows_writable_sids(path) - allowed
    except OSError as exc:
        raise StorageSecurityError(
            "STORAGE_UNAVAILABLE", "could not inspect the storage ACL"
        ) from exc
    if unexpected:
        raise StorageSecurityError(
            "STORAGE_UNAVAILABLE",
            "storage is writable by a principal other than the current user",
        )


def _check_posix(path: Path, *, private: bool) -> None:
    info = os.lstat(path)
    if info.st_uid != os.geteuid():
        raise StorageSecurityError(
            "STORAGE_UNAVAILABLE", "storage is not owned by the current user"
        )
    permissions = stat.S_IMODE(info.st_mode)
    if (private and permissions & 0o077) or (not private and permissions & 0o002):
        raise StorageSecurityError(
            "STORAGE_UNAVAILABLE", "storage permissions are too broad"
        )


def resolve_storage_root(path: Path, *, create: bool = False) -> Path:
    """Resolve once and reject root links, nondirectories, and unsafe writers."""

    candidate = Path(path).expanduser().absolute()
    if os.path.lexists(candidate) and is_reparse_point(candidate):
        raise StorageSecurityError("STORAGE_UNAVAILABLE", "storage root cannot be a link")
    if not candidate.exists():
        if not create:
            raise StorageSecurityError("STORAGE_UNAVAILABLE", "storage root does not exist")
        candidate.mkdir(mode=0o700, parents=True, exist_ok=False)
        if os.name == "nt":
            _set_windows_private_acl(candidate)
        else:
            os.chmod(candidate, 0o700)
        _fsync_directory(candidate.parent)
    if not candidate.is_dir() or is_reparse_point(candidate):
        raise StorageSecurityError(
            "STORAGE_UNAVAILABLE", "storage root is not a safe directory"
        )
    resolved = candidate.resolve(strict=True)
    if os.name == "nt":
        _check_windows_private_acl(resolved)
    else:
        _check_posix(resolved, private=False)
    return resolved


def ensure_private_directory(path: Path) -> None:
    target = Path(path).absolute()
    _reject_linked_components(target, include_leaf=True)
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    _reject_linked_components(target, include_leaf=True)
    if not target.is_dir():
        raise StorageSecurityError("STORAGE_UNAVAILABLE", "private directory is unsafe")
    if os.name == "nt":
        _set_windows_private_acl(target)
        _check_windows_private_acl(target)
    else:
        os.chmod(target, 0o700)
        _check_posix(target, private=True)


def check_private_file(path: Path) -> None:
    target = Path(path)
    _reject_linked_components(target, include_leaf=True)
    if not target.is_file():
        raise StorageSecurityError("STORAGE_UNAVAILABLE", "private file is unsafe")
    if os.name == "nt":
        _check_windows_private_acl(target)
    else:
        info = os.lstat(target)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o177:
            raise StorageSecurityError(
                "STORAGE_UNAVAILABLE", "private file permissions are too broad"
            )


def _write_all(stream: BinaryIO, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = stream.write(view)
        if written is None or written <= 0:
            raise OSError("short write")
        view = view[written:]


def write_private_file(path: Path, data: bytes) -> None:
    """Atomically replace a 0600/current-user-only file beside its target."""

    target = Path(path).absolute()
    ensure_private_directory(target.parent)
    _reject_linked_components(target, include_leaf=True)
    descriptor, name = tempfile.mkstemp(prefix=".private-", dir=target.parent)
    temporary = Path(name)
    try:
        if os.name != "nt":
            os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            _write_all(stream, data)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "nt":
            _set_windows_private_acl(temporary)
        os.replace(temporary, target)
        if os.name == "nt":
            _set_windows_private_acl(target)
        else:
            os.chmod(target, 0o600)
        _fsync_directory(target.parent)
        check_private_file(target)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


class InstanceLease:
    """Exclusive root lease; workers may explicitly inherit its descriptor."""

    def __init__(self, root: Path) -> None:
        self.root = resolve_storage_root(root)
        self.runtime_dir = self.root / "runtime"
        self.path = self.runtime_dir / "instance.lock"
        self._stream: BinaryIO | None = None

    @property
    def fileno(self) -> int:
        if self._stream is None:
            raise RuntimeError("instance lease is not acquired")
        return self._stream.fileno()

    def acquire(self) -> "InstanceLease":
        if self._stream is not None:
            return self
        ensure_private_directory(self.runtime_dir)
        _reject_linked_components(self.path, include_leaf=True)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags, 0o600)
        stream = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            _reject_linked_components(self.path, include_leaf=True)
            descriptor_info = os.fstat(stream.fileno())
            if not stat.S_ISREG(descriptor_info.st_mode):
                raise StorageSecurityError(
                    "STORAGE_UNAVAILABLE", "instance lease is not a regular file"
                )
            if os.name != "nt":
                path_info = os.lstat(self.path)
                if (path_info.st_dev, path_info.st_ino) != (
                    descriptor_info.st_dev,
                    descriptor_info.st_ino,
                ):
                    raise StorageSecurityError(
                        "STORAGE_UNAVAILABLE", "instance lease changed during open"
                    )
            if os.name == "nt":
                import msvcrt

                if self.path.stat().st_size == 0:
                    stream.write(b"0")
                    stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                _set_windows_private_acl(self.path)
                check_private_file(self.path)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.fchmod(stream.fileno(), 0o600)
            os.set_inheritable(stream.fileno(), False)
            stream.seek(0)
            stream.truncate(0)
            stream.write(f"{os.getpid()}\n".encode("ascii"))
            stream.flush()
            os.fsync(stream.fileno())
            self._stream = stream
            return self
        except StorageSecurityError:
            stream.close()
            raise
        except (BlockingIOError, OSError) as exc:
            stream.close()
            raise StorageSecurityError(
                "STORAGE_IN_USE", "storage root is already in use"
            ) from exc

    def make_inheritable(self) -> int:
        os.set_inheritable(self.fileno, True)
        return self.fileno

    def release(self) -> None:
        if self._stream is None:
            return
        stream, self._stream = self._stream, None
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def __enter__(self) -> "InstanceLease":
        return self.acquire()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()
