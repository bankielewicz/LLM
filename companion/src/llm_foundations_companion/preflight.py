"""Bounded runtime profile selection and startup preflight orchestration.

This service-process module deliberately uses only the Python standard
library. Backend imports are confined to ``preflight_main``, executed in a
fixed isolated child.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence


Profile = Literal["win-cpu", "win-cuda", "wsl-cpu", "wsl-cuda"]
PlatformKind = Literal["windows", "wsl"]
Device = Literal["cpu", "cuda"]

PROFILES: tuple[Profile, ...] = (
    "win-cpu",
    "win-cuda",
    "wsl-cpu",
    "wsl-cuda",
)
CPU_TORCH_VERSION = "2.8.0+cpu"
CUDA_TORCH_VERSION = "2.8.0+cu128"
MINIMUM_DRIVER = (572, 61)
MINIMUM_COMPUTE_CAPABILITY = (7, 5)
MINIMUM_TOTAL_MEMORY_MIB = 3_584
MINIMUM_FREE_MEMORY_MIB = 2_048
MAX_RELATIVE_DIFFERENCE = 1e-5
GPU_QUERY_TIMEOUT_SECONDS = 10.0
GPU_QUERY_STDOUT_LIMIT = 4_096
PREFLIGHT_CHILD_TIMEOUT_SECONDS = 30.0
PREFLIGHT_CHILD_STREAM_LIMIT = 65_536
PREFLIGHT_CHILD_FORMAT = "llm-foundations-preflight-child-v1"
PREFLIGHT_RECEIPT_FORMAT = "llm-foundations-preflight-v1"

GPU_QUERY_ARGS: tuple[str, ...] = (
    "--query-gpu=name,compute_cap,driver_version,memory.total",
    "--format=csv,noheader,nounits",
)
GPU_QUERY_EXECUTABLES: Mapping[PlatformKind, str] = MappingProxyType(
    {
        "windows": r"C:\Windows\System32\nvidia-smi.exe",
        "wsl": "/usr/lib/wsl/lib/nvidia-smi",
    }
)
BACKEND_VERSION_PINS: Mapping[str, str] = MappingProxyType(
    {
        "transformers": "4.57.1",
        "peft": "0.17.1",
        "accelerate": "1.10.1",
        "safetensors": "0.6.2",
    }
)

_OPERATION_NAMES = (
    "tokenizer_train",
    "tiny_train",
    "tiny_resume",
    "evaluate",
    "generate",
    "model_prepare",
    "adapter_train",
    "adapter_resume",
    "context_preview",
    "chat_generate",
    "retrieval_build",
    "retrieval_query",
    "export_bundle",
    "validate_bundle",
    "import_bundle",
)
S2_OPERATION_NAMES = frozenset({"tokenizer_train", "tiny_train", "tiny_resume", "evaluate"})
_UNAVAILABLE_MESSAGE = "This operation is not available in this implementation slice."
OPERATION_CAPABILITIES: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        operation: MappingProxyType(
            {"available": True} if operation in S2_OPERATION_NAMES else {
                "available": False,
                "reason_code": "CAPABILITY_UNAVAILABLE",
                "message": _UNAVAILABLE_MESSAGE,
            }
        )
        for operation in _OPERATION_NAMES
    }
)

_COMPUTE_CAPABILITY_RE = re.compile(r"^[0-9]{1,2}\.[0-9]$")
_DRIVER_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")
_ARCH_RE = re.compile(r"^(?:sm|compute)_([0-9]{2,3})[a-z]?$")
_PYTHON_VERSION_RE = re.compile(r"^3\.12\.[0-9]+$")
_CHILD_VERSION_KEYS = frozenset(
    {"torch", "transformers", "peft", "accelerate", "safetensors"}
)
_CHILD_CUDA_REASONS = frozenset(
    {
        "CUDA_DEVICE_NOT_FOUND",
        "CUDA_CAPABILITY_UNSUPPORTED",
        "CUDA_ARCH_NOT_IN_BUILD",
        "CUDA_MEMORY_INSUFFICIENT",
        "CUDA_NUMERIC_CHECK_FAILED",
    }
)


class PreflightError(RuntimeError):
    """Stable startup failure without exception or host-path disclosure."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message


class UnsupportedPlatformError(PreflightError):
    """The host is outside the two release operating-system families."""


class ProfileSelectionError(PreflightError):
    """An explicit profile cannot be used by the running environment."""

    def __init__(
        self, reason_code: str, message: str, *, cpu_command: str | None = None
    ) -> None:
        super().__init__(reason_code, message)
        self.cpu_command = cpu_command


class CapabilityUnavailable(PreflightError):
    """A job operation is unavailable before scheduler insertion."""

    def __init__(self, operation: str, reason_code: str, message: str) -> None:
        super().__init__(reason_code, message)
        self.operation = operation


@dataclass(frozen=True)
class ProfileSelection:
    profile: Profile
    platform_kind: PlatformKind
    device: Device
    explicit_profile: bool
    observed_torch_version: str | None
    backend_reason_code: str | None


@dataclass(frozen=True)
class GpuQueryLine:
    name: str
    compute_capability: str
    driver_version: str
    memory_total_mib: int

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "compute_capability": self.compute_capability,
            "driver_version": self.driver_version,
            "memory_total_mib": self.memory_total_mib,
        }


@dataclass(frozen=True)
class ProcessResult:
    returncode: int | None
    stdout: bytes = b""
    stderr: bytes = b""
    timed_out: bool = False
    stdout_exceeded: bool = False
    stderr_exceeded: bool = False
    launch_error: bool = False


class _WindowsJobObject:
    """Retained kill-on-close container for one owned Windows process tree."""

    def __init__(
        self,
        handle: int,
        assign_process: Callable[[int, int], int],
        close_handle: Callable[[int], int],
    ) -> None:
        self._handle: int | None = handle
        self._assign_process = assign_process
        self._close_handle = close_handle

    def assign(self, process: subprocess.Popen[bytes]) -> None:
        process_handle = getattr(process, "_handle", None)
        if process_handle is None or not self._assign_process(
            self._required_handle(), int(process_handle)
        ):
            raise OSError("The preflight child could not be assigned to its Job Object.")

    def close(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None:
            self._close_handle(handle)

    def _required_handle(self) -> int:
        if self._handle is None:
            raise OSError("The preflight child Job Object is already closed.")
        return self._handle


def _create_windows_kill_on_close_job() -> _WindowsJobObject:
    """Create the RUN-009 Windows container without a third-party dependency."""

    import ctypes

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("per_process_user_time_limit", ctypes.c_int64),
            ("per_job_user_time_limit", ctypes.c_int64),
            ("limit_flags", ctypes.c_uint32),
            ("minimum_working_set_size", ctypes.c_size_t),
            ("maximum_working_set_size", ctypes.c_size_t),
            ("active_process_limit", ctypes.c_uint32),
            ("affinity", ctypes.c_size_t),
            ("priority_class", ctypes.c_uint32),
            ("scheduling_class", ctypes.c_uint32),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("read_operation_count", ctypes.c_uint64),
            ("write_operation_count", ctypes.c_uint64),
            ("other_operation_count", ctypes.c_uint64),
            ("read_transfer_count", ctypes.c_uint64),
            ("write_transfer_count", ctypes.c_uint64),
            ("other_transfer_count", ctypes.c_uint64),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("basic_limit_information", BasicLimitInformation),
            ("io_info", IoCounters),
            ("process_memory_limit", ctypes.c_size_t),
            ("job_memory_limit", ctypes.c_size_t),
            ("peak_process_memory_used", ctypes.c_size_t),
            ("peak_job_memory_used", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_job = kernel32.CreateJobObjectW
    create_job.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    create_job.restype = ctypes.c_void_p
    set_information = kernel32.SetInformationJobObject
    set_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    set_information.restype = ctypes.c_int
    assign_process = kernel32.AssignProcessToJobObject
    assign_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    assign_process.restype = ctypes.c_int
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int

    handle = create_job(None, None)
    if not handle:
        raise OSError("The preflight child Job Object could not be created.")
    information = ExtendedLimitInformation()
    information.basic_limit_information.limit_flags = 0x00002000
    if not set_information(
        handle, 9, ctypes.byref(information), ctypes.sizeof(information)
    ):
        close_handle(handle)
        raise OSError("The preflight child Job Object could not be configured.")
    return _WindowsJobObject(int(handle), assign_process, close_handle)


@dataclass(frozen=True)
class GpuQueryResult:
    gpu: GpuQueryLine | None
    query_line: str | None
    reason_code: str | None
    process_result: ProcessResult


@dataclass(frozen=True)
class ChildExecution:
    report: Mapping[str, object]
    stderr_text: str
    process_result: ProcessResult
    valid_child_output: bool


@dataclass(frozen=True)
class StorageProbeResult:
    writable: bool
    write_ok: bool
    read_digest_ok: bool
    delete_ok: bool
    write_observed: str
    read_observed: str
    delete_observed: str


class ProcessRunner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        stdout_limit: int,
        stderr_limit: int,
        env: Mapping[str, str],
        cwd: str | None,
    ) -> ProcessResult: ...


def detect_platform(
    *,
    system_name: str | None = None,
    release: str | None = None,
    version: str | None = None,
    machine: str | None = None,
) -> PlatformKind:
    """Return ``windows`` or ``wsl``; native Linux is deliberately rejected."""

    system_name = system_name if system_name is not None else platform.system()
    release = release if release is not None else platform.release()
    version = version if version is not None else platform.version()
    machine = machine if machine is not None else platform.machine()
    if machine.casefold() not in {"amd64", "x86_64"}:
        raise UnsupportedPlatformError(
            "BACKEND_VERSION_MISMATCH",
            "The companion release profiles require an x86_64 host.",
        )
    if system_name.casefold() == "windows":
        return "windows"
    kernel_identity = f"{release} {version}".casefold()
    if system_name.casefold() == "linux" and "microsoft" in kernel_identity:
        return "wsl"
    raise UnsupportedPlatformError(
        "BACKEND_VERSION_MISMATCH",
        "The companion release profiles support native Windows 11 or Ubuntu under WSL2.",
    )


def _profile_for_build(platform_kind: PlatformKind, torch_version: str) -> Profile | None:
    prefix = "win" if platform_kind == "windows" else "wsl"
    if torch_version == CPU_TORCH_VERSION:
        return f"{prefix}-cpu"  # type: ignore[return-value]
    if torch_version == CUDA_TORCH_VERSION:
        return f"{prefix}-cuda"  # type: ignore[return-value]
    return None


def _cpu_profile(platform_kind: PlatformKind) -> Profile:
    return "win-cpu" if platform_kind == "windows" else "wsl-cpu"


def _cuda_profile(platform_kind: PlatformKind) -> Profile:
    return "win-cuda" if platform_kind == "windows" else "wsl-cuda"


def _read_torch_version(
    resolver: Callable[[str], str],
) -> tuple[str | None, str | None]:
    try:
        return resolver("torch"), None
    except importlib.metadata.PackageNotFoundError:
        return None, "BACKEND_MISSING"
    except Exception:
        return None, "BACKEND_MISSING"


def select_profile(
    explicit_profile: str | None,
    *,
    platform_kind: PlatformKind | None = None,
    torch_version_resolver: Callable[[str], str] = importlib.metadata.version,
) -> ProfileSelection:
    """Select from installed Torch metadata without importing a backend."""

    platform_kind = platform_kind or detect_platform()
    if platform_kind not in ("windows", "wsl"):
        raise UnsupportedPlatformError(
            "BACKEND_VERSION_MISMATCH", "Unsupported runtime platform."
        )
    allowed = {
        "windows": frozenset({"win-cpu", "win-cuda"}),
        "wsl": frozenset({"wsl-cpu", "wsl-cuda"}),
    }[platform_kind]
    if explicit_profile is not None and explicit_profile not in PROFILES:
        raise ProfileSelectionError(
            "BACKEND_VERSION_MISMATCH", "Unknown runtime profile."
        )
    if explicit_profile is not None and explicit_profile not in allowed:
        raise ProfileSelectionError(
            "BACKEND_VERSION_MISMATCH",
            "The explicit runtime profile does not match this operating system.",
        )

    torch_version, missing_reason = _read_torch_version(torch_version_resolver)
    matched = (
        _profile_for_build(platform_kind, torch_version)
        if torch_version is not None
        else None
    )
    reason = missing_reason
    if torch_version is not None and matched is None:
        reason = "BACKEND_VERSION_MISMATCH"

    if explicit_profile is not None:
        if matched != explicit_profile:
            raise ProfileSelectionError(
                "BACKEND_VERSION_MISMATCH",
                "The explicit runtime profile does not match the installed Torch build.",
            )
        profile_name: Profile = explicit_profile  # type: ignore[assignment]
        return ProfileSelection(
            profile=profile_name,
            platform_kind=platform_kind,
            device="cuda" if profile_name.endswith("-cuda") else "cpu",
            explicit_profile=True,
            observed_torch_version=torch_version,
            backend_reason_code=None,
        )

    profile_name = matched or _cpu_profile(platform_kind)
    return ProfileSelection(
        profile=profile_name,
        platform_kind=platform_kind,
        device="cuda" if profile_name.endswith("-cuda") else "cpu",
        explicit_profile=False,
        observed_torch_version=torch_version,
        backend_reason_code=reason,
    )


def parse_gpu_query_line(line: str) -> GpuQueryLine:
    """Parse GPU 0 from the four-field fixed ``nvidia-smi`` query."""

    if not isinstance(line, str) or not line or len(line) > 400:
        raise ValueError("GPU query line is missing or too long")
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 4:
        raise ValueError("GPU query line must have four comma-separated fields")
    name, capability, driver, memory_text = fields
    if not name or len(name) > 200:
        raise ValueError("GPU name is invalid")
    if not _COMPUTE_CAPABILITY_RE.fullmatch(capability):
        raise ValueError("GPU compute capability is invalid")
    if not _DRIVER_RE.fullmatch(driver):
        raise ValueError("GPU driver version is invalid")
    if not re.fullmatch(r"[0-9]+", memory_text):
        raise ValueError("GPU memory is invalid")
    return GpuQueryLine(
        name=name,
        compute_capability=capability,
        driver_version=driver,
        memory_total_mib=int(memory_text),
    )


def _version_tuple(value: str) -> tuple[int, ...]:
    if not _DRIVER_RE.fullmatch(value):
        raise ValueError("invalid numeric dotted version")
    return tuple(int(component) for component in value.split("."))


def _version_at_least(value: str, minimum: tuple[int, ...]) -> bool:
    parsed = _version_tuple(value)
    size = max(len(parsed), len(minimum))
    return parsed + (0,) * (size - len(parsed)) >= minimum + (0,) * (
        size - len(minimum)
    )


def _capability_tuple(value: str) -> tuple[int, int]:
    if not _COMPUTE_CAPABILITY_RE.fullmatch(value):
        raise ValueError("invalid compute capability")
    major, minor = value.split(".")
    return int(major), int(minor)


def run_gpu_query(
    platform_kind: PlatformKind,
    *,
    process_runner: ProcessRunner | None = None,
) -> GpuQueryResult:
    """Run the fixed GPU query with no caller-controlled executable or args."""

    if platform_kind not in GPU_QUERY_EXECUTABLES:
        raise UnsupportedPlatformError(
            "BACKEND_VERSION_MISMATCH", "Unsupported runtime platform."
        )
    runner = process_runner or _run_fixed_process
    argv = (GPU_QUERY_EXECUTABLES[platform_kind], *GPU_QUERY_ARGS)
    result = runner(
        argv,
        timeout_seconds=GPU_QUERY_TIMEOUT_SECONDS,
        stdout_limit=GPU_QUERY_STDOUT_LIMIT,
        stderr_limit=GPU_QUERY_STDOUT_LIMIT,
        env=_fixed_process_environment(platform_kind, profile_name=None),
        cwd=None,
    )
    if (
        result.launch_error
        or result.timed_out
        or result.stdout_exceeded
        or result.stderr_exceeded
        or len(result.stdout) > GPU_QUERY_STDOUT_LIMIT
        or result.returncode != 0
    ):
        return GpuQueryResult(None, None, "CUDA_DEVICE_NOT_FOUND", result)
    try:
        text = result.stdout.decode("utf-8", errors="strict")
        lines = text.splitlines()
        first_line = lines[0] if lines else ""
        parsed = parse_gpu_query_line(first_line)
    except (UnicodeDecodeError, ValueError, IndexError):
        return GpuQueryResult(None, None, "CUDA_DEVICE_NOT_FOUND", result)
    return GpuQueryResult(parsed, first_line, None, result)


def detect_cuda_environment_installed(
    selection: ProfileSelection,
    *,
    interpreter: str | os.PathLike[str] | None = None,
) -> bool:
    """Check the sibling CUDA interpreter only for an exact envs/<profile> run."""

    interpreter_path = Path(interpreter or sys.executable)
    environment = interpreter_path.parent.parent
    if environment.name != selection.profile or environment.parent.name != "envs":
        return False
    cuda_environment = environment.parent / _cuda_profile(selection.platform_kind)
    relative = (
        Path("Scripts") / "python.exe"
        if selection.platform_kind == "windows"
        else Path("bin") / "python"
    )
    return (cuda_environment / relative).is_file()


def gpu_offer(
    selection: ProfileSelection,
    query: GpuQueryResult,
    *,
    cuda_environment_installed: bool,
) -> dict[str, object]:
    """Build the exact RuntimeInfo ``GpuOffer`` shape in required order."""

    reason: str | None = None
    gpu = query.gpu
    if selection.device == "cuda":
        if gpu is None:
            raise PreflightError(
                "CUDA_DEVICE_NOT_FOUND",
                "A CUDA instance requires a successful fixed GPU query.",
            )
        status = "in_use"
    elif gpu is None:
        status = "not_detected"
    elif not _version_at_least(gpu.driver_version, MINIMUM_DRIVER):
        status = "unsupported"
        reason = "CUDA_DRIVER_TOO_OLD"
    elif _capability_tuple(gpu.compute_capability) < MINIMUM_COMPUTE_CAPABILITY:
        status = "unsupported"
        reason = "CUDA_CAPABILITY_UNSUPPORTED"
    elif gpu.memory_total_mib < MINIMUM_TOTAL_MEMORY_MIB:
        status = "unsupported"
        reason = "CUDA_MEMORY_INSUFFICIENT"
    else:
        status = "available"
    return {
        "status": status,
        "gpu": gpu.as_dict() if gpu is not None else None,
        "cuda_profile": _cuda_profile(selection.platform_kind),
        "reason_code": reason,
        "cuda_environment_installed": bool(cuda_environment_installed),
        "explicit_profile": selection.explicit_profile,
    }


def gpu_check_message(offer: Mapping[str, object]) -> str:
    """Return the fixed RUN-001 sentence for a GPU offer."""

    status = offer.get("status")
    gpu_value = offer.get("gpu")
    gpu = gpu_value if isinstance(gpu_value, Mapping) else {}
    name = str(gpu.get("name", ""))
    cuda_profile = str(offer.get("cuda_profile", ""))
    if status == "not_detected":
        return "No NVIDIA GPU found; the CPU environment runs every lesson."
    if status == "available":
        if offer.get("cuda_environment_installed") is True:
            return (
                f"Supported GPU found: {name}. The {cuda_profile} environment is "
                "installed; start the companion from it to use the GPU."
            )
        return (
            f"Supported GPU found: {name}. Recommended: also install the "
            f"{cuda_profile} environment (setup guide, GPU section); module 17's "
            "device comparison uses both environments."
        )
    if status == "unsupported":
        reason = offer.get("reason_code")
        if reason == "CUDA_DRIVER_TOO_OLD":
            sentence = (
                f"its driver is {gpu.get('driver_version')} and 572.61 or newer is "
                "required."
            )
        elif reason == "CUDA_CAPABILITY_UNSUPPORTED":
            sentence = (
                f"its compute capability is {gpu.get('compute_capability')} and 7.5 "
                "or newer (GeForce RTX 20-series or later) is required."
            )
        elif reason == "CUDA_MEMORY_INSUFFICIENT":
            sentence = (
                f"it has {gpu.get('memory_total_mib')} MiB of memory and 3,584 MiB "
                "is required."
            )
        else:
            raise ValueError("unsupported offer has no recognized reason")
        return f"GPU found but not supported: {sentence}"
    if status == "in_use":
        return f"Supported GPU found: {name}. The companion is using the GPU."
    raise ValueError("unknown GPU offer status")


def gpu_check(
    *,
    platform_kind: PlatformKind | None = None,
    process_runner: ProcessRunner | None = None,
    interpreter: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, object], str]:
    """Return the GPU offer and fixed sentence without reading or importing Torch."""

    platform_kind = platform_kind or detect_platform()
    cpu_profile = _cpu_profile(platform_kind)
    cpu_selection = ProfileSelection(
        profile=cpu_profile,
        platform_kind=platform_kind,
        device="cpu",
        explicit_profile=False,
        observed_torch_version=None,
        backend_reason_code=None,
    )
    query = run_gpu_query(platform_kind, process_runner=process_runner)
    offer = gpu_offer(
        cpu_selection,
        query,
        cuda_environment_installed=detect_cuda_environment_installed(
            cpu_selection, interpreter=interpreter
        ),
    )
    return offer, gpu_check_message(offer)


def suggested_cpu_command(
    selection: ProfileSelection,
    *,
    storage_root_display: str = "<storage-root>",
) -> str:
    """Return the fixed display command for the OS CPU environment."""

    cpu_profile = _cpu_profile(selection.platform_kind)
    if selection.platform_kind == "windows":
        interpreter = rf"envs\{cpu_profile}\Scripts\python.exe"
    else:
        interpreter = f"envs/{cpu_profile}/bin/python"
    return (
        f"{interpreter} -m llm_foundations_companion serve --storage "
        f"{storage_root_display}"
    )


def require_startup_allowed(
    selection: ProfileSelection,
    child: ChildExecution,
    query: GpuQueryResult,
    *,
    storage_root_display: str = "<storage-root>",
) -> dict[str, object] | None:
    """Reject a CUDA startup before root/lock publication; permit CPU readers."""

    if selection.device == "cpu":
        return None
    command = suggested_cpu_command(
        selection, storage_root_display=storage_root_display
    )
    if query.gpu is None:
        raise ProfileSelectionError(
            "CUDA_DEVICE_NOT_FOUND",
            f"The fixed GPU query found no supported NVIDIA GPU. Start the CPU profile with: {command}",
            cpu_command=command,
        )
    if not _version_at_least(query.gpu.driver_version, MINIMUM_DRIVER):
        raise ProfileSelectionError(
            "CUDA_DRIVER_TOO_OLD",
            f"The Windows NVIDIA driver must be 572.61 or newer. Start the CPU profile with: {command}",
            cpu_command=command,
        )
    backend_reason = _backend_reason(selection, child.report)
    if backend_reason is not None:
        meaning = (
            "Required backend imports are unavailable."
            if backend_reason == "BACKEND_MISSING"
            else "Installed backend versions differ from the locked runtime profile."
        )
        raise ProfileSelectionError(
            backend_reason,
            f"{meaning} Start the CPU profile with: {command}",
            cpu_command=command,
        )
    cuda = evaluate_cuda_preflight(query, child.report)
    if cuda["result"] != "passed":
        reason = str(cuda["reason_code"])
        raise ProfileSelectionError(
            reason,
            f"CUDA startup preflight failed with {reason}. Start the CPU profile with: {command}",
            cpu_command=command,
        )
    return cuda


def _fixed_process_environment(
    platform_kind: PlatformKind,
    *,
    profile_name: Profile | None,
) -> dict[str, str]:
    environment = {
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    if profile_name is not None:
        environment["LLM_FOUNDATIONS_PROFILE"] = profile_name
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = "0"
    if platform_kind == "windows":
        for name in ("SystemRoot", "WINDIR"):
            value = os.environ.get(name)
            if value:
                environment[name] = value
    else:
        environment.update({"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
    return environment


def _terminate_owned_process(
    process: subprocess.Popen[bytes],
    windows_job: _WindowsJobObject | None,
) -> None:
    try:
        if windows_job is not None:
            windows_job.close()
        elif os.name == "nt":
            if process.poll() is None:
                process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass


def _bounded_reader(
    stream: Any,
    limit: int,
    buffer: bytearray,
    exceeded: threading.Event,
) -> None:
    try:
        while True:
            chunk = stream.read(8_192)
            if not chunk:
                break
            remaining = max(0, limit + 1 - len(buffer))
            if remaining:
                buffer.extend(chunk[:remaining])
            if len(buffer) > limit or len(chunk) > remaining:
                exceeded.set()
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except (OSError, ValueError):
            pass


def _run_fixed_process(
    argv: Sequence[str],
    *,
    timeout_seconds: float,
    stdout_limit: int,
    stderr_limit: int,
    env: Mapping[str, str],
    cwd: str | None,
) -> ProcessResult:
    """Run one owned, shell-free process with bounded capture and deadline."""

    deadline = time.monotonic() + timeout_seconds
    creationflags = 0
    popen_kwargs: dict[str, object] = {}
    windows_job: _WindowsJobObject | None = None
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            windows_job = _create_windows_kill_on_close_job()
        except OSError:
            return ProcessResult(returncode=None, launch_error=True)
    else:
        popen_kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(
            tuple(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            cwd=cwd,
            env=dict(env),
            creationflags=creationflags,
            **popen_kwargs,
        )
    except OSError:
        if windows_job is not None:
            windows_job.close()
        return ProcessResult(returncode=None, launch_error=True)
    if windows_job is not None:
        try:
            windows_job.assign(process)
        except OSError:
            try:
                process.kill()
                process.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
            windows_job.close()
            return ProcessResult(returncode=None, launch_error=True)
    assert process.stdout is not None and process.stderr is not None
    stdout = bytearray()
    stderr = bytearray()
    stdout_exceeded = threading.Event()
    stderr_exceeded = threading.Event()
    threads = (
        threading.Thread(
            target=_bounded_reader,
            args=(process.stdout, stdout_limit, stdout, stdout_exceeded),
            daemon=True,
        ),
        threading.Thread(
            target=_bounded_reader,
            args=(process.stderr, stderr_limit, stderr, stderr_exceeded),
            daemon=True,
        ),
    )
    for thread in threads:
        thread.start()
    completed = threading.Event()
    completion: list[tuple[int | None, float]] = []

    def wait_for_process() -> None:
        try:
            returncode: int | None = process.wait()
        except OSError:
            returncode = None
        completion.append((returncode, time.monotonic()))
        completed.set()

    waiter = threading.Thread(target=wait_for_process, daemon=True)
    waiter.start()
    timed_out = False
    while True:
        if stdout_exceeded.is_set() or stderr_exceeded.is_set():
            break
        if completed.is_set():
            timed_out = bool(completion and completion[0][1] > deadline)
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if completed.is_set():
                timed_out = bool(completion and completion[0][1] > deadline)
            else:
                timed_out = True
            break
        completed.wait(min(remaining, 0.01))
    _terminate_owned_process(process, windows_job)
    if not completed.wait(2.0):
        timed_out = True
        _terminate_owned_process(process, windows_job)
    waiter.join(timeout=2.0)
    for thread in threads:
        thread.join(timeout=2.0)
    for thread, stream in zip(threads, (process.stdout, process.stderr), strict=True):
        if thread.is_alive():
            try:
                stream.close()
            except (OSError, ValueError):
                pass
            thread.join(timeout=2.0)
    returncode = completion[0][0] if completion else process.poll()
    if returncode is None and not timed_out:
        timed_out = True
    return ProcessResult(
        returncode=returncode,
        stdout=bytes(stdout[:stdout_limit]),
        stderr=bytes(stderr[:stderr_limit]),
        timed_out=timed_out,
        stdout_exceeded=stdout_exceeded.is_set(),
        stderr_exceeded=stderr_exceeded.is_set(),
    )


def _closed_keys(value: object, expected: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} is not a closed object")
    return value


def _validate_child_device(value: object) -> None:
    device = _closed_keys(
        value,
        frozenset(
            {
                "name",
                "count",
                "compute_capability",
                "arch_list",
                "total_memory_mib",
                "free_memory_mib",
                "max_relative_difference",
            }
        ),
        "child cuda device",
    )
    if not isinstance(device["name"], str) or not 1 <= len(device["name"]) <= 200:
        raise ValueError("child CUDA device name is invalid")
    if type(device["count"]) is not int or not 1 <= device["count"] <= 64:
        raise ValueError("child CUDA device count is invalid")
    capability = device["compute_capability"]
    if not isinstance(capability, str) or not _COMPUTE_CAPABILITY_RE.fullmatch(capability):
        raise ValueError("child CUDA compute capability is invalid")
    arch_list = device["arch_list"]
    if not isinstance(arch_list, list) or len(arch_list) > 32:
        raise ValueError("child CUDA architecture list is invalid")
    if any(not isinstance(item, str) or not _ARCH_RE.fullmatch(item) for item in arch_list):
        raise ValueError("child CUDA architecture entry is invalid")
    for key in ("total_memory_mib", "free_memory_mib"):
        if type(device[key]) is not int or device[key] < 0:
            raise ValueError(f"child CUDA {key} is invalid")
    difference = device["max_relative_difference"]
    if difference is not None and (
        isinstance(difference, bool)
        or not isinstance(difference, (int, float))
        or not math.isfinite(float(difference))
        or difference < 0
    ):
        raise ValueError("child CUDA relative difference is invalid")


def validate_child_report(value: object) -> Mapping[str, object]:
    """Validate the closed child contract without importing jsonschema."""

    report = _closed_keys(
        value,
        frozenset(
            {
                "format",
                "backend_status",
                "device_available",
                "versions",
                "error_text",
                "cuda",
            }
        ),
        "preflight child report",
    )
    if report["format"] != PREFLIGHT_CHILD_FORMAT:
        raise ValueError("preflight child format is invalid")
    if (
        not isinstance(report["backend_status"], str)
        or report["backend_status"] not in {"passed", "missing", "version_mismatch"}
    ):
        raise ValueError("preflight child backend status is invalid")
    if type(report["device_available"]) is not bool:
        raise ValueError("preflight child device availability is invalid")
    versions = _closed_keys(report["versions"], _CHILD_VERSION_KEYS, "child versions")
    for value_item in versions.values():
        if value_item is not None and (
            not isinstance(value_item, str) or len(value_item) > 80
        ):
            raise ValueError("preflight child version is invalid")
    error_text = report["error_text"]
    if not isinstance(error_text, str) or len(error_text) > 2_000:
        raise ValueError("preflight child error text is invalid")
    cuda = report["cuda"]
    if cuda is not None:
        cuda_value = _closed_keys(
            cuda,
            frozenset({"result", "reason_code", "device"}),
            "child cuda result",
        )
        if (
            not isinstance(cuda_value["result"], str)
            or cuda_value["result"] not in {"passed", "failed"}
        ):
            raise ValueError("child CUDA result is invalid")
        reason = cuda_value["reason_code"]
        if reason is not None and (
            not isinstance(reason, str) or reason not in _CHILD_CUDA_REASONS
        ):
            raise ValueError("child CUDA reason is invalid")
        if cuda_value["device"] is not None:
            _validate_child_device(cuda_value["device"])
        if cuda_value["result"] == "passed":
            if reason is not None or cuda_value["device"] is None:
                raise ValueError("passing child CUDA result is incomplete")
            device = cuda_value["device"]
            assert isinstance(device, Mapping)
            if device["max_relative_difference"] is None:
                raise ValueError("passing child CUDA result lacks numeric check")
        elif reason is None:
            raise ValueError("failed child CUDA result lacks a reason")
    return report


def _synthetic_child_report(
    selection: ProfileSelection,
    *,
    backend_status: Literal["missing", "version_mismatch"],
    error_text: str,
) -> dict[str, object]:
    return {
        "format": PREFLIGHT_CHILD_FORMAT,
        "backend_status": backend_status,
        "device_available": False,
        "versions": {name: None for name in sorted(_CHILD_VERSION_KEYS)},
        "error_text": error_text[:2_000],
        "cuda": (
            {
                "result": "failed",
                "reason_code": "CUDA_DEVICE_NOT_FOUND",
                "device": None,
            }
            if selection.device == "cuda"
            else None
        ),
    }


def _parse_child_stdout(stdout: bytes) -> Mapping[str, object]:
    text = stdout.decode("utf-8", errors="strict")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(token: str) -> Any:
        raise ValueError(f"non-finite JSON number: {token}")

    value = json.loads(
        text,
        object_pairs_hook=reject_duplicate,
        parse_constant=reject_constant,
    )
    return validate_child_report(value)


def run_preflight_child(
    selection: ProfileSelection,
    *,
    interpreter: str | os.PathLike[str] | None = None,
    process_runner: ProcessRunner | None = None,
) -> ChildExecution:
    """Execute exactly one fixed isolated backend probe."""

    interpreter_text = os.fspath(interpreter or sys.executable)
    runner = process_runner or _run_fixed_process
    result = runner(
        (interpreter_text, "-I", "-m", "llm_foundations_companion.preflight_main"),
        timeout_seconds=PREFLIGHT_CHILD_TIMEOUT_SECONDS,
        stdout_limit=PREFLIGHT_CHILD_STREAM_LIMIT,
        stderr_limit=PREFLIGHT_CHILD_STREAM_LIMIT,
        env=_fixed_process_environment(
            selection.platform_kind, profile_name=selection.profile
        ),
        cwd=str(Path(interpreter_text).parent),
    )
    stderr_text = result.stderr.decode("utf-8", errors="replace")
    if (
        result.launch_error
        or result.timed_out
        or result.stdout_exceeded
        or result.stderr_exceeded
        or len(result.stdout) > PREFLIGHT_CHILD_STREAM_LIMIT
        or len(result.stderr) > PREFLIGHT_CHILD_STREAM_LIMIT
        or result.returncode != 0
    ):
        report = _synthetic_child_report(
            selection,
            backend_status="missing",
            error_text="The bounded backend preflight child did not complete successfully.",
        )
        return ChildExecution(report, stderr_text, result, False)
    try:
        report = _parse_child_stdout(result.stdout)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        report = _synthetic_child_report(
            selection,
            backend_status="missing",
            error_text="The bounded backend preflight child returned an invalid report.",
        )
        return ChildExecution(report, stderr_text, result, False)
    return ChildExecution(report, stderr_text, result, True)


def _compiled_architecture_supports(
    arch_list: Sequence[object], capability: tuple[int, int]
) -> bool:
    major, minor = capability
    for entry in arch_list:
        if not isinstance(entry, str) or not entry.startswith("sm_"):
            continue
        match = _ARCH_RE.fullmatch(entry)
        if not match:
            continue
        digits = match.group(1)
        if int(digits[:-1]) == major and int(digits[-1]) <= minor:
            return True
    return False


def evaluate_cuda_preflight(
    query: GpuQueryResult,
    child_report: Mapping[str, object],
) -> dict[str, object]:
    """Apply the seven CUDA decisions in normative first-failure order."""

    if query.gpu is None:
        return {
            "result": "failed",
            "reason_code": "CUDA_DEVICE_NOT_FOUND",
            "query_line": query.query_line,
            "device": None,
        }
    if not _version_at_least(query.gpu.driver_version, MINIMUM_DRIVER):
        return {
            "result": "failed",
            "reason_code": "CUDA_DRIVER_TOO_OLD",
            "query_line": query.query_line,
            "device": None,
        }
    validate_child_report(child_report)
    child_cuda_value = child_report.get("cuda")
    child_cuda = child_cuda_value if isinstance(child_cuda_value, Mapping) else None
    if child_report.get("device_available") is not True or child_cuda is None:
        return {
            "result": "failed",
            "reason_code": "CUDA_DEVICE_NOT_FOUND",
            "query_line": query.query_line,
            "device": None,
        }
    device_value = child_cuda.get("device")
    if not isinstance(device_value, Mapping):
        reason = child_cuda.get("reason_code")
        if reason not in _CHILD_CUDA_REASONS:
            reason = "CUDA_DEVICE_NOT_FOUND"
        return {
            "result": "failed",
            "reason_code": reason,
            "query_line": query.query_line,
            "device": None,
        }
    device = dict(device_value)
    capability = _capability_tuple(str(device["compute_capability"]))
    if capability < MINIMUM_COMPUTE_CAPABILITY:
        reason = "CUDA_CAPABILITY_UNSUPPORTED"
    elif not _compiled_architecture_supports(device["arch_list"], capability):
        reason = "CUDA_ARCH_NOT_IN_BUILD"
    elif (
        int(device["total_memory_mib"]) < MINIMUM_TOTAL_MEMORY_MIB
        or int(device["free_memory_mib"]) < MINIMUM_FREE_MEMORY_MIB
    ):
        reason = "CUDA_MEMORY_INSUFFICIENT"
    else:
        difference = device["max_relative_difference"]
        if difference is None or float(difference) > MAX_RELATIVE_DIFFERENCE:
            reason = "CUDA_NUMERIC_CHECK_FAILED"
        else:
            reason = None
    return {
        "result": "failed" if reason else "passed",
        "reason_code": reason,
        "query_line": query.query_line,
        "device": device,
    }


def _backend_reason(
    selection: ProfileSelection, child_report: Mapping[str, object]
) -> str | None:
    if selection.backend_reason_code is not None:
        return selection.backend_reason_code
    status = child_report.get("backend_status")
    if status == "missing":
        return "BACKEND_MISSING"
    if status == "version_mismatch":
        return "BACKEND_VERSION_MISMATCH"
    return None


def capabilities_from_preflight(
    selection: ProfileSelection,
    child: ChildExecution,
) -> dict[str, dict[str, object]]:
    """Return closed RuntimeInfo capability records for the current slice."""

    reason = _backend_reason(selection, child.report)
    if reason is None:
        return {name: dict(record) for name, record in OPERATION_CAPABILITIES.items()}
    message = (
        "Required backend imports are unavailable."
        if reason == "BACKEND_MISSING"
        else "Installed backend versions differ from the locked runtime profile."
    )
    return {
        name: {"available": False, "reason_code": reason, "message": message}
        for name in OPERATION_CAPABILITIES
    }


def capability_enabled(
    capabilities: Mapping[str, Mapping[str, object]], operation: str
) -> bool:
    record = capabilities.get(operation)
    return bool(record is not None and record.get("available") is True)


def require_capability(
    capabilities: Mapping[str, Mapping[str, object]], operation: str
) -> None:
    record = capabilities.get(operation)
    if record is not None and record.get("available") is True:
        return
    reason = (
        str(record.get("reason_code"))
        if record is not None and record.get("reason_code")
        else "CAPABILITY_UNAVAILABLE"
    )
    message = (
        str(record.get("message"))
        if record is not None and record.get("message")
        else "The requested operation is unavailable."
    )
    raise CapabilityUnavailable(operation, reason, message)


def _test_row(identifier: str, passed: bool, observed: str) -> dict[str, str]:
    return {
        "id": identifier,
        "status": "pass" if passed else "fail",
        "observed": observed[:2_000],
    }


def _canonical_receipt_digest(receipt: Mapping[str, object]) -> str:
    digest_input = {
        key: value
        for key, value in receipt.items()
        if key not in {"receipt_artifact_id", "receipt_sha256"}
    }
    encoded = json.dumps(
        digest_input,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_preflight_receipt(
    selection: ProfileSelection,
    child: ChildExecution,
    storage: StorageProbeResult,
    *,
    storage_root_display: str,
    started_at: str,
    finished_at: str,
    gpu_query: GpuQueryResult | None = None,
    python_version: str | None = None,
    interpreter_label: str | None = None,
    receipt_artifact_id: str | None = None,
) -> dict[str, object]:
    """Combine injected storage evidence with the bounded child report."""

    python_version = python_version or platform.python_version()
    interpreter_label = interpreter_label or (
        f"{platform.python_implementation()} {python_version}"
    )
    backend_reason = _backend_reason(selection, child.report)
    backend_passed = backend_reason is None
    cuda: dict[str, object] | None = None
    profile_reason = backend_reason
    if selection.device == "cuda" and profile_reason is None:
        if gpu_query is None:
            raise ValueError("CUDA receipt requires a GPU query result")
        cuda = evaluate_cuda_preflight(gpu_query, child.report)
        if cuda["result"] != "passed":
            profile_reason = str(cuda["reason_code"])
    elif selection.device == "cpu" and child.report.get("device_available") is not True:
        profile_reason = profile_reason or "BACKEND_MISSING"

    tests = [
        _test_row(
            "P00-RUN-001-PYTHON",
            bool(_PYTHON_VERSION_RE.fullmatch(python_version)),
            f"Python {python_version}",
        ),
        _test_row(
            "P00-RUN-002-PROFILE",
            profile_reason is None,
            profile_reason or f"{selection.profile} uses {selection.device}",
        ),
        _test_row(
            "P00-RUN-003-STORAGE-WRITE",
            storage.write_ok,
            storage.write_observed,
        ),
        _test_row(
            "P00-RUN-004-STORAGE-READ-DIGEST",
            storage.read_digest_ok,
            storage.read_observed,
        ),
        _test_row(
            "P00-RUN-005-STORAGE-DELETE",
            storage.delete_ok,
            storage.delete_observed,
        ),
        _test_row(
            "P00-RUN-006-BACKEND-CAPABILITIES",
            backend_passed,
            backend_reason or "Locked backend imports and versions passed.",
        ),
    ]
    receipt: dict[str, object] = {
        "format": PREFLIGHT_RECEIPT_FORMAT,
        "receipt_artifact_id": receipt_artifact_id,
        "python_version": python_version,
        "interpreter_label": interpreter_label[:120],
        "profile": selection.profile,
        "device": selection.device,
        "storage_root_display": storage_root_display[:500],
        "storage_writable": storage.writable,
        "tests": tests,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": "pass" if all(row["status"] == "pass" for row in tests) else "fail",
        "receipt_sha256": "",
        "cuda": cuda,
    }
    receipt["receipt_sha256"] = _canonical_receipt_digest(receipt)
    return receipt


__all__ = [
    "CapabilityUnavailable",
    "ChildExecution",
    "GpuQueryLine",
    "GpuQueryResult",
    "OPERATION_CAPABILITIES",
    "PreflightError",
    "ProcessResult",
    "ProfileSelection",
    "ProfileSelectionError",
    "StorageProbeResult",
    "UnsupportedPlatformError",
    "build_preflight_receipt",
    "capabilities_from_preflight",
    "capability_enabled",
    "detect_cuda_environment_installed",
    "detect_platform",
    "evaluate_cuda_preflight",
    "gpu_check",
    "gpu_check_message",
    "gpu_offer",
    "parse_gpu_query_line",
    "require_capability",
    "require_startup_allowed",
    "run_gpu_query",
    "run_preflight_child",
    "select_profile",
    "suggested_cpu_command",
    "validate_child_report",
]
