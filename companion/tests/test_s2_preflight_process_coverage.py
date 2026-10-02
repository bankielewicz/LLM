"""Process-control failure coverage with explicit in-process doubles.

These tests exercise preflight containment logic only. They do not launch the
shared service and do not establish native Windows runtime qualification.
"""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

import pytest

from llm_foundations_companion import preflight


class _Function:
    def __init__(self, callback):
        self.callback = callback
        self.calls: list[tuple[object, ...]] = []
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        self.calls.append(args)
        return self.callback(*args)


class _Structure:
    _fields_: list[tuple[str, object]] = []

    def __init__(self) -> None:
        for name, field_type in self._fields_:
            value = field_type() if isinstance(field_type, type) and issubclass(field_type, _Structure) else 0
            setattr(self, name, value)


def _fake_ctypes(*, create_result=41, configure_result=1):
    create_job = _Function(lambda *_: create_result)
    set_information = _Function(lambda *_: configure_result)
    assign_process = _Function(lambda *_: 1)
    close_handle = _Function(lambda *_: 1)
    kernel32 = SimpleNamespace(
        CreateJobObjectW=create_job,
        SetInformationJobObject=set_information,
        AssignProcessToJobObject=assign_process,
        CloseHandle=close_handle,
    )
    scalar = object()
    module = SimpleNamespace(
        Structure=_Structure,
        c_int64=scalar,
        c_uint32=scalar,
        c_size_t=scalar,
        c_uint64=scalar,
        c_void_p=scalar,
        c_wchar_p=scalar,
        c_int=scalar,
        WinDLL=lambda *_args, **_kwargs: kernel32,
        byref=lambda value: value,
        sizeof=lambda _value: 256,
    )
    return module, kernel32


def test_windows_job_object_assign_close_and_closed_handle_contract() -> None:
    assigned: list[tuple[int, int]] = []
    closed: list[int] = []
    job = preflight._WindowsJobObject(
        41,
        lambda job_handle, process_handle: assigned.append((job_handle, process_handle)) or 1,
        lambda handle: closed.append(handle) or 1,
    )
    job.assign(SimpleNamespace(_handle=73))
    job.close()
    job.close()
    assert assigned == [(41, 73)]
    assert closed == [41]
    with pytest.raises(OSError, match="already closed"):
        job._required_handle()
    with pytest.raises(OSError, match="could not be assigned"):
        job.assign(SimpleNamespace())


def test_windows_job_object_rejects_failed_assignment() -> None:
    job = preflight._WindowsJobObject(41, lambda *_: 0, lambda *_: 1)
    with pytest.raises(OSError, match="could not be assigned"):
        job.assign(SimpleNamespace(_handle=73))


def test_windows_job_factory_configures_kill_on_close(monkeypatch) -> None:
    ctypes, kernel32 = _fake_ctypes()
    monkeypatch.setitem(sys.modules, "ctypes", ctypes)
    job = preflight._create_windows_kill_on_close_job()
    information = kernel32.SetInformationJobObject.calls[0][2]
    assert kernel32.CreateJobObjectW.calls == [(None, None)]
    assert kernel32.SetInformationJobObject.calls[0][0:2] == (41, 9)
    assert information.basic_limit_information.limit_flags == 0x00002000
    assert kernel32.SetInformationJobObject.calls[0][3] == 256
    job.assign(SimpleNamespace(_handle=73))
    job.close()
    assert kernel32.AssignProcessToJobObject.calls == [(41, 73)]
    assert kernel32.CloseHandle.calls == [(41,)]


@pytest.mark.parametrize(("create_result", "configure_result", "closed"), [(0, 1, False), (41, 0, True)])
def test_windows_job_factory_reports_creation_and_configuration_failures(monkeypatch, create_result, configure_result, closed) -> None:
    ctypes, kernel32 = _fake_ctypes(create_result=create_result, configure_result=configure_result)
    monkeypatch.setitem(sys.modules, "ctypes", ctypes)
    with pytest.raises(OSError, match="could not be (?:created|configured)"):
        preflight._create_windows_kill_on_close_job()
    assert bool(kernel32.CloseHandle.calls) is closed


class _Stream:
    def __init__(self, chunks, *, read_error=False, close_error=False) -> None:
        self.chunks = iter(chunks)
        self.read_error = read_error
        self.close_error = close_error
        self.closed = False

    def read(self, _size):
        if self.read_error:
            raise OSError("synthetic read failure")
        return next(self.chunks, b"")

    def close(self):
        self.closed = True
        if self.close_error:
            raise ValueError("synthetic close failure")


@pytest.mark.parametrize(("stream", "expected", "exceeded"), [(_Stream([b"abcdef"]), b"abcd", True), (_Stream([], read_error=True, close_error=True), b"", False)])
def test_bounded_reader_contains_stream_and_close_failures(stream, expected, exceeded) -> None:
    buffer = bytearray()
    overflow = threading.Event()
    preflight._bounded_reader(stream, 4, buffer, overflow)
    assert bytes(buffer[:4]) == expected
    assert overflow.is_set() is exceeded
    assert stream.closed is True


class _Process:
    def __init__(self, *, wait_error=False, release=None, kill_error=False) -> None:
        self._handle = 73
        self.pid = 12345
        self.stdout = _Stream([])
        self.stderr = _Stream([])
        self.wait_error = wait_error
        self.release = release
        self.kill_error = kill_error
        self.killed = 0
        self.wait_calls: list[object] = []

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self.wait_error:
            raise OSError("synthetic wait failure")
        if self.release is not None:
            self.release.wait()
        return -9 if self.killed else 0

    def kill(self):
        if self.kill_error:
            raise OSError("synthetic kill failure")
        self.killed += 1
        if self.release is not None:
            self.release.set()

    def poll(self):
        return None


class _Job:
    def __init__(self, *, assign_error=False, close_action=None) -> None:
        self.assign_error = assign_error
        self.close_action = close_action
        self.assigned: list[object] = []
        self.closed = 0

    def assign(self, process):
        self.assigned.append(process)
        if self.assign_error:
            raise OSError("synthetic assignment failure")

    def close(self):
        self.closed += 1
        if self.close_action is not None:
            self.close_action()


def _windows(monkeypatch, *, job, popen):
    monkeypatch.setattr(preflight.os, "name", "nt")
    monkeypatch.setattr(preflight.subprocess, "CREATE_NEW_PROCESS_GROUP", 512, raising=False)
    monkeypatch.setattr(preflight, "_create_windows_kill_on_close_job", lambda: job)
    monkeypatch.setattr(preflight.subprocess, "Popen", popen)


def _run(*, timeout_seconds=0.05):
    return preflight._run_fixed_process(
        ("fixed-python", "-I", "-m", "fixed-child"),
        timeout_seconds=timeout_seconds,
        stdout_limit=8,
        stderr_limit=8,
        env={"PYTHONUTF8": "1"},
        cwd="fixed-cwd",
    )


def test_windows_job_creation_failure_prevents_process_launch(monkeypatch) -> None:
    launched = []
    monkeypatch.setattr(preflight.os, "name", "nt")
    monkeypatch.setattr(preflight.subprocess, "CREATE_NEW_PROCESS_GROUP", 512, raising=False)
    monkeypatch.setattr(preflight, "_create_windows_kill_on_close_job", lambda: (_ for _ in ()).throw(OSError("synthetic job failure")))
    monkeypatch.setattr(preflight.subprocess, "Popen", lambda *a, **k: launched.append((a, k)))
    assert _run() == preflight.ProcessResult(returncode=None, launch_error=True)
    assert launched == []


def test_windows_launch_failure_closes_owned_job(monkeypatch) -> None:
    job = _Job()
    _windows(monkeypatch, job=job, popen=lambda *a, **k: (_ for _ in ()).throw(OSError("synthetic spawn failure")))
    assert _run() == preflight.ProcessResult(returncode=None, launch_error=True)
    assert job.closed == 1


def test_windows_assignment_failure_kills_child_and_closes_job(monkeypatch) -> None:
    process = _Process()
    job = _Job(assign_error=True)
    _windows(monkeypatch, job=job, popen=lambda *a, **k: process)
    assert _run() == preflight.ProcessResult(returncode=None, launch_error=True)
    assert job.assigned == [process]
    assert process.killed == 1 and process.wait_calls == [2.0]
    assert job.closed == 1


def test_windows_assignment_cleanup_failure_is_contained(monkeypatch) -> None:
    process = _Process(kill_error=True)
    job = _Job(assign_error=True)
    _windows(monkeypatch, job=job, popen=lambda *a, **k: process)
    assert _run() == preflight.ProcessResult(returncode=None, launch_error=True)
    assert process.killed == 0 and process.wait_calls == []
    assert job.closed == 1


def test_windows_wait_failure_is_bounded_as_timeout(monkeypatch) -> None:
    process = _Process(wait_error=True)
    job = _Job()
    _windows(monkeypatch, job=job, popen=lambda *a, **k: process)
    result = _run()
    assert result.returncode is None and result.timed_out is True
    assert job.closed == 1


def test_windows_deadline_closes_job_and_reaps_owned_child(monkeypatch) -> None:
    release = threading.Event()
    process = _Process(release=release)
    job = _Job(close_action=release.set)
    _windows(monkeypatch, job=job, popen=lambda *a, **k: process)
    result = _run()
    assert result.timed_out is True and result.returncode == 0
    assert job.closed == 1 and release.is_set()


def test_termination_failure_falls_back_to_direct_child_kill(monkeypatch) -> None:
    process = _Process()
    monkeypatch.setattr(preflight.os, "killpg", lambda *_: (_ for _ in ()).throw(ProcessLookupError("synthetic missing group")))
    preflight._terminate_owned_process(process, None)
    assert process.killed == 1


def test_windows_direct_child_termination_without_job(monkeypatch) -> None:
    process = _Process()
    monkeypatch.setattr(preflight.os, "name", "nt")
    preflight._terminate_owned_process(process, None)
    assert process.killed == 1


def test_termination_suppresses_fallback_kill_failure(monkeypatch) -> None:
    process = _Process(kill_error=True)
    monkeypatch.setattr(preflight.os, "killpg", lambda *_: (_ for _ in ()).throw(ProcessLookupError("synthetic missing group")))
    preflight._terminate_owned_process(process, None)
    assert process.killed == 0


def test_windows_process_environment_copies_only_required_host_roots(monkeypatch) -> None:
    monkeypatch.setenv("SystemRoot", "C:/SyntheticWindows")
    monkeypatch.delenv("WINDIR", raising=False)
    environment = preflight._fixed_process_environment("windows", profile_name="win-cpu")
    assert environment == {
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "LLM_FOUNDATIONS_PROFILE": "win-cpu",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0",
        "SystemRoot": "C:/SyntheticWindows",
    }


def test_deadline_completion_race_uses_observed_completion_time(monkeypatch) -> None:
    real_event = threading.Event
    release = real_event()
    process = _Process(release=release)
    job = _Job()

    class RaceEvent:
        def __init__(self) -> None:
            self.flag = real_event()
            self.first = True

        def set(self):
            self.flag.set()

        def is_set(self):
            if self.first:
                self.first = False
                release.set()
                assert self.flag.wait(1.0)
                return False
            return self.flag.is_set()

        def wait(self, timeout=None):
            return self.flag.wait(timeout)

    events = iter((real_event(), real_event(), RaceEvent()))
    monkeypatch.setattr(
        preflight,
        "threading",
        SimpleNamespace(Event=lambda: next(events), Thread=threading.Thread),
    )
    _windows(monkeypatch, job=job, popen=lambda *a, **k: process)
    result = _run(timeout_seconds=0.0)
    assert result.timed_out is True and result.returncode == 0


def test_unresponsive_waiter_gets_second_owned_termination(monkeypatch) -> None:
    real_event = threading.Event
    release = real_event()
    process = _Process(release=release)
    job = _Job()
    job.close_action = lambda: release.set() if job.closed == 2 else None

    class ImmediateTimeoutEvent:
        def __init__(self) -> None:
            self.flag = real_event()

        def set(self):
            self.flag.set()

        def is_set(self):
            return self.flag.is_set()

        def wait(self, timeout=None):
            return False

    events = iter((real_event(), real_event(), ImmediateTimeoutEvent()))
    monkeypatch.setattr(
        preflight,
        "threading",
        SimpleNamespace(Event=lambda: next(events), Thread=threading.Thread),
    )
    _windows(monkeypatch, job=job, popen=lambda *a, **k: process)
    result = _run(timeout_seconds=0.0)
    assert result.timed_out is True
    assert job.closed == 2 and release.is_set()


def test_lingering_reader_streams_are_closed_after_process_exit(monkeypatch) -> None:
    process = _Process()
    process.stderr = _Stream([], close_error=True)
    job = _Job()

    class ControlledThread:
        def __init__(self, *, target, args=(), daemon=False):
            self.target = target
            self.args = args
            self.alive = False

        def start(self):
            if self.target.__name__ == "wait_for_process":
                self.target(*self.args)
            else:
                self.alive = True

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return self.alive

    monkeypatch.setattr(preflight.threading, "Thread", ControlledThread)
    _windows(monkeypatch, job=job, popen=lambda *a, **k: process)
    result = _run()
    assert result.returncode == 0 and result.timed_out is False
    assert process.stdout.closed and process.stderr.closed
