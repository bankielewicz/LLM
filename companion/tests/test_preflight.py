from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from llm_foundations_companion import preflight
from llm_foundations_companion import preflight_main


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "docs" / "specs" / "intermediate-v1"


def resolver(version: str):
    return lambda name: version


def missing_resolver(name: str) -> str:
    raise importlib.metadata.PackageNotFoundError(name)


def selection(
    profile: preflight.Profile = "wsl-cpu",
    *,
    explicit: bool = False,
    backend_reason: str | None = None,
) -> preflight.ProfileSelection:
    return preflight.ProfileSelection(
        profile=profile,
        platform_kind="windows" if profile.startswith("win-") else "wsl",
        device="cuda" if profile.endswith("-cuda") else "cpu",
        explicit_profile=explicit,
        observed_torch_version=(
            preflight.CUDA_TORCH_VERSION
            if profile.endswith("-cuda")
            else preflight.CPU_TORCH_VERSION
        ),
        backend_reason_code=backend_reason,
    )


def passing_cpu_report() -> dict[str, object]:
    return {
        "format": "llm-foundations-preflight-child-v1",
        "backend_status": "passed",
        "device_available": True,
        "versions": {
            "torch": "2.8.0+cpu",
            "transformers": "4.57.1",
            "peft": "0.17.1",
            "accelerate": "1.10.1",
            "safetensors": "0.6.2",
        },
        "error_text": "",
        "cuda": None,
    }


def cuda_device(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "name": "NVIDIA GeForce RTX 5070",
        "count": 1,
        "compute_capability": "12.0",
        "arch_list": ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"],
        "total_memory_mib": 12_227,
        "free_memory_mib": 11_000,
        "max_relative_difference": 2e-7,
    }
    value.update(changes)
    return value


def cuda_report(
    *,
    device_available: bool = True,
    device: dict[str, object] | None = None,
    result: str = "passed",
    reason_code: str | None = None,
) -> dict[str, object]:
    if device is None and device_available:
        device = cuda_device()
    return {
        "format": "llm-foundations-preflight-child-v1",
        "backend_status": "passed",
        "device_available": device_available,
        "versions": {
            "torch": "2.8.0+cu128",
            "transformers": "4.57.1",
            "peft": "0.17.1",
            "accelerate": "1.10.1",
            "safetensors": "0.6.2",
        },
        "error_text": "",
        "cuda": {
            "result": result,
            "reason_code": reason_code,
            "device": device,
        },
    }


def gpu_result(
    line: str = "NVIDIA GeForce RTX 5070, 12.0, 616.92, 12227",
) -> preflight.GpuQueryResult:
    return preflight.GpuQueryResult(
        gpu=preflight.parse_gpu_query_line(line),
        query_line=line,
        reason_code=None,
        process_result=preflight.ProcessResult(returncode=0, stdout=(line + "\n").encode()),
    )


def child_execution(
    report: dict[str, object] | None = None,
) -> preflight.ChildExecution:
    return preflight.ChildExecution(
        report=report or passing_cpu_report(),
        stderr_text="",
        process_result=preflight.ProcessResult(returncode=0),
        valid_child_output=True,
    )


def storage_probe() -> preflight.StorageProbeResult:
    return preflight.StorageProbeResult(
        writable=True,
        write_ok=True,
        read_digest_ok=True,
        delete_ok=True,
        write_observed="4,096 random bytes flushed to service staging.",
        read_observed="Read-back SHA-256 matched.",
        delete_observed="Probe deleted and absence verified.",
    )


def openapi_validator(schema_name: str) -> Draft202012Validator:
    document = json.loads((SPEC / "contracts" / "openapi.json").read_text(encoding="utf-8"))
    uri = "urn:llm-foundations:openapi"
    resource = Resource.from_contents(document, default_specification=DRAFT202012)
    registry = Registry().with_resource(uri, resource)
    return Draft202012Validator(
        {"$ref": f"{uri}#/components/schemas/{schema_name}"}, registry=registry
    )


def assert_valid(validator: Draft202012Validator, value: object) -> None:
    errors = sorted(validator.iter_errors(value), key=lambda item: list(item.path))
    assert errors == []


def test_service_preflight_imports_no_ml_backend() -> None:
    source = ROOT / "companion" / "src" / "llm_foundations_companion" / "preflight.py"
    code = """\
import importlib.util
import sys

names = {"torch", "transformers", "peft"}
spec = importlib.util.spec_from_file_location("_llmf_preflight_probe", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
loaded = sorted(names & set(sys.modules))
if loaded:
    raise SystemExit("ML backend imported: " + ", ".join(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(source)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"system_name": "Windows", "release": "11", "version": "11", "machine": "AMD64"}, "windows"),
        ({"system_name": "Linux", "release": "6.8.0-microsoft-standard-WSL2", "version": "Linux", "machine": "x86_64"}, "wsl"),
    ],
)
def test_platform_detection_accepts_only_release_families(
    arguments: dict[str, str], expected: str
) -> None:
    assert preflight.detect_platform(**arguments) == expected


@pytest.mark.parametrize(
    "arguments",
    [
        {"system_name": "Linux", "release": "6.8.0-generic", "version": "Linux", "machine": "x86_64"},
        {"system_name": "Darwin", "release": "25", "version": "Darwin", "machine": "arm64"},
    ],
)
def test_platform_detection_rejects_native_linux_and_other_hosts(
    arguments: dict[str, str]
) -> None:
    with pytest.raises(preflight.UnsupportedPlatformError) as raised:
        preflight.detect_platform(**arguments)
    assert raised.value.reason_code == "BACKEND_VERSION_MISMATCH"


@pytest.mark.parametrize(
    ("platform_kind", "version", "expected"),
    [
        ("windows", "2.8.0+cpu", "win-cpu"),
        ("windows", "2.8.0+cu128", "win-cuda"),
        ("wsl", "2.8.0+cpu", "wsl-cpu"),
        ("wsl", "2.8.0+cu128", "wsl-cuda"),
    ],
)
def test_implicit_profile_matches_exact_torch_metadata(
    platform_kind: preflight.PlatformKind, version: str, expected: str
) -> None:
    chosen = preflight.select_profile(
        None,
        platform_kind=platform_kind,
        torch_version_resolver=resolver(version),
    )
    assert chosen.profile == expected
    assert chosen.backend_reason_code is None


def test_implicit_missing_and_wrong_torch_choose_cpu_with_unavailable_backend() -> None:
    missing = preflight.select_profile(
        None, platform_kind="wsl", torch_version_resolver=missing_resolver
    )
    wrong = preflight.select_profile(
        None,
        platform_kind="windows",
        torch_version_resolver=resolver("2.8.0"),
    )
    assert (missing.profile, missing.backend_reason_code) == (
        "wsl-cpu",
        "BACKEND_MISSING",
    )
    assert (wrong.profile, wrong.backend_reason_code) == (
        "win-cpu",
        "BACKEND_VERSION_MISMATCH",
    )


@pytest.mark.parametrize(
    ("explicit", "platform_kind", "version"),
    [
        ("wsl-cuda", "wsl", "2.8.0+cpu"),
        ("win-cpu", "wsl", "2.8.0+cpu"),
        ("unknown", "wsl", "2.8.0+cpu"),
    ],
)
def test_explicit_profile_mismatch_fails_before_lock(
    explicit: str, platform_kind: preflight.PlatformKind, version: str
) -> None:
    with pytest.raises(preflight.ProfileSelectionError) as raised:
        preflight.select_profile(
            explicit,
            platform_kind=platform_kind,
            torch_version_resolver=resolver(version),
        )
    assert raised.value.reason_code == "BACKEND_VERSION_MISMATCH"


def test_profile_selection_is_immutable() -> None:
    chosen = selection()
    with pytest.raises(FrozenInstanceError):
        chosen.profile = "wsl-cuda"  # type: ignore[misc]


def test_gpu_parser_accepts_only_the_closed_four_field_line() -> None:
    parsed = preflight.parse_gpu_query_line(
        " NVIDIA GeForce RTX 5070 , 12.0 , 616.92 , 12227 "
    )
    assert parsed.as_dict() == {
        "name": "NVIDIA GeForce RTX 5070",
        "compute_capability": "12.0",
        "driver_version": "616.92",
        "memory_total_mib": 12227,
    }
    for malformed in (
        "",
        "GPU, 12.0, 616.92",
        "GPU, 12, 616.92, 12227",
        "GPU, 12.0, 616, 12227",
        "GPU, 12.0, 616.92, 12 GiB",
    ):
        with pytest.raises(ValueError):
            preflight.parse_gpu_query_line(malformed)


def test_gpu_query_uses_only_fixed_executable_args_and_limits() -> None:
    captured: dict[str, Any] = {}

    def runner(argv, **kwargs):
        captured.update(argv=tuple(argv), **kwargs)
        return preflight.ProcessResult(
            returncode=0,
            stdout=b"NVIDIA GeForce RTX 5070, 12.0, 616.92, 12227\n",
        )

    result = preflight.run_gpu_query("wsl", process_runner=runner)
    assert result.gpu is not None
    assert captured["argv"] == (
        "/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,compute_cap,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    )
    assert captured["timeout_seconds"] == 10.0
    assert captured["stdout_limit"] == 4096
    assert "HTTP_PROXY" not in captured["env"]


@pytest.mark.parametrize(
    "result",
    [
        preflight.ProcessResult(returncode=None, launch_error=True),
        preflight.ProcessResult(returncode=9),
        preflight.ProcessResult(returncode=0, timed_out=True),
        preflight.ProcessResult(returncode=0, stdout_exceeded=True),
        preflight.ProcessResult(returncode=0, stdout=b"malformed\n"),
        preflight.ProcessResult(
            returncode=0,
            stdout=b"GPU, 7.5, 572.61, 3584\n" + b"x" * 4_097,
        ),
    ],
)
def test_gpu_query_failure_modes_are_not_detected(
    result: preflight.ProcessResult,
) -> None:
    query = preflight.run_gpu_query("wsl", process_runner=lambda *args, **kwargs: result)
    assert query.gpu is None
    assert query.reason_code == "CUDA_DEVICE_NOT_FOUND"


@pytest.mark.parametrize(
    ("line", "status", "reason"),
    [
        ("GPU, 12.0, 572.60, 12000", "unsupported", "CUDA_DRIVER_TOO_OLD"),
        ("GPU, 7.4, 572.61, 12000", "unsupported", "CUDA_CAPABILITY_UNSUPPORTED"),
        ("GPU, 7.5, 572.61, 3583", "unsupported", "CUDA_MEMORY_INSUFFICIENT"),
        ("GPU, 7.5, 572.61, 3584", "available", None),
    ],
)
def test_gpu_offer_hardware_boundaries(
    line: str, status: str, reason: str | None
) -> None:
    offer = preflight.gpu_offer(
        selection(), gpu_result(line), cuda_environment_installed=False
    )
    assert (offer["status"], offer["reason_code"]) == (status, reason)
    assert_valid(openapi_validator("GpuOffer"), offer)


def test_gpu_offer_not_detected_and_in_use_are_schema_valid() -> None:
    missing = preflight.GpuQueryResult(
        None,
        None,
        "CUDA_DEVICE_NOT_FOUND",
        preflight.ProcessResult(returncode=None, launch_error=True),
    )
    not_detected = preflight.gpu_offer(
        selection(), missing, cuda_environment_installed=False
    )
    in_use = preflight.gpu_offer(
        selection("wsl-cuda", explicit=True),
        gpu_result(),
        cuda_environment_installed=True,
    )
    assert not_detected["status"] == "not_detected"
    assert in_use["status"] == "in_use"
    validator = openapi_validator("GpuOffer")
    assert_valid(validator, not_detected)
    assert_valid(validator, in_use)


def test_gpu_check_sentences_are_exact() -> None:
    available = preflight.gpu_offer(
        selection(), gpu_result(), cuda_environment_installed=False
    )
    assert preflight.gpu_check_message(available) == (
        "Supported GPU found: NVIDIA GeForce RTX 5070. Recommended: also install "
        "the wsl-cuda environment (setup guide, GPU section); module 17's device "
        "comparison uses both environments."
    )


def test_gpu_check_runs_query_only_and_returns_offer_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preflight.importlib.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(AssertionError("metadata read")),
    )
    offer, sentence = preflight.gpu_check(
        platform_kind="wsl",
        interpreter="/not-an-installed-env/bin/python",
        process_runner=lambda *args, **kwargs: preflight.ProcessResult(
            returncode=0,
            stdout=b"NVIDIA GeForce RTX 5070, 12.0, 616.92, 12227\n",
        ),
    )
    assert offer["status"] == "available"
    assert offer["cuda_environment_installed"] is False
    assert sentence.startswith("Supported GPU found: NVIDIA GeForce RTX 5070.")


def test_child_invocation_is_fixed_isolated_and_allowlisted() -> None:
    captured: dict[str, Any] = {}
    encoded = json.dumps(passing_cpu_report(), separators=(",", ":")).encode()

    def runner(argv, **kwargs):
        captured.update(argv=tuple(argv), **kwargs)
        return preflight.ProcessResult(returncode=0, stdout=encoded)

    result = preflight.run_preflight_child(
        selection(), interpreter="/fixed/env/bin/python", process_runner=runner
    )
    assert result.valid_child_output is True
    assert captured["argv"] == (
        "/fixed/env/bin/python",
        "-I",
        "-m",
        "llm_foundations_companion.preflight_main",
    )
    assert captured["timeout_seconds"] == 30.0
    assert captured["stdout_limit"] == 65536
    assert captured["env"] == {
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "LLM_FOUNDATIONS_PROFILE": "wsl-cpu",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }


@pytest.mark.parametrize(
    "process_result",
    [
        preflight.ProcessResult(returncode=9),
        preflight.ProcessResult(returncode=0, timed_out=True),
        preflight.ProcessResult(returncode=0, stdout_exceeded=True),
        preflight.ProcessResult(returncode=0, stderr_exceeded=True),
        preflight.ProcessResult(returncode=0, stdout=b"{}"),
        preflight.ProcessResult(returncode=0, stdout=b'{"format":NaN}'),
        preflight.ProcessResult(
            returncode=0,
            stdout=json.dumps(passing_cpu_report(), separators=(",", ":")).encode()
            + b" " * 65_537,
        ),
    ],
)
def test_child_failures_become_bounded_backend_missing(
    process_result: preflight.ProcessResult,
) -> None:
    result = preflight.run_preflight_child(
        selection(), process_runner=lambda *args, **kwargs: process_result
    )
    assert result.valid_child_output is False
    assert result.report["backend_status"] == "missing"
    assert result.report["device_available"] is False


@pytest.mark.skipif(os.name == "nt", reason="WSL process-group behavior")
@pytest.mark.parametrize("limit", [4_096, 65_536])
def test_real_process_runner_enforces_exact_stream_boundaries(limit: int) -> None:
    exact = preflight._run_fixed_process(
        (
            sys.executable,
            "-c",
            f"import os; os.write(1, b'x' * {limit})",
        ),
        timeout_seconds=2.0,
        stdout_limit=limit,
        stderr_limit=limit,
        env={"PYTHONUTF8": "1"},
        cwd=None,
    )
    above = preflight._run_fixed_process(
        (
            sys.executable,
            "-c",
            f"import os; os.write(1, b'x' * {limit + 1})",
        ),
        timeout_seconds=2.0,
        stdout_limit=limit,
        stderr_limit=limit,
        env={"PYTHONUTF8": "1"},
        cwd=None,
    )
    assert exact.returncode == 0
    assert len(exact.stdout) == limit
    assert exact.stdout_exceeded is False
    assert len(above.stdout) == limit
    assert above.stdout_exceeded is True


@pytest.mark.skipif(os.name == "nt", reason="WSL process-group behavior")
def test_real_process_runner_enforces_deadline() -> None:
    started = time.monotonic()
    result = preflight._run_fixed_process(
        (sys.executable, "-c", "import time; time.sleep(10)"),
        timeout_seconds=0.05,
        stdout_limit=4_096,
        stderr_limit=4_096,
        env={"PYTHONUTF8": "1"},
        cwd=None,
    )
    elapsed = time.monotonic() - started
    assert result.timed_out is True
    assert result.returncode is not None
    assert elapsed < 3.0


@pytest.mark.skipif(os.name == "nt", reason="WSL process-group behavior")
def test_real_process_runner_kills_only_its_owned_group(tmp_path: Path) -> None:
    spawned = tmp_path / "spawned"
    survivor = tmp_path / "survivor"
    grandchild = (
        "import pathlib,time; time.sleep(0.8); "
        f"pathlib.Path({str(survivor)!r}).write_text('survived')"
    )
    parent = (
        "import pathlib,subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
        f"pathlib.Path({str(spawned)!r}).write_text('spawned'); "
        "time.sleep(10)"
    )
    result = preflight._run_fixed_process(
        (sys.executable, "-c", parent),
        timeout_seconds=0.4,
        stdout_limit=4_096,
        stderr_limit=4_096,
        env={"PYTHONUTF8": "1"},
        cwd=None,
    )
    assert result.timed_out is True
    assert spawned.is_file()
    time.sleep(1.0)
    assert not survivor.exists()


def test_real_process_runner_returns_launch_errors_as_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_spawn(*args: Any, **kwargs: Any) -> Any:
        raise OSError("synthetic spawn failure")

    monkeypatch.setattr(preflight.subprocess, "Popen", failed_spawn)
    result = preflight._run_fixed_process(
        (sys.executable, "-c", "pass"),
        timeout_seconds=2.0,
        stdout_limit=4_096,
        stderr_limit=4_096,
        env={"PYTHONUTF8": "1"},
        cwd=None,
    )
    assert result == preflight.ProcessResult(returncode=None, launch_error=True)


def test_child_report_is_closed_and_matches_authoritative_schema() -> None:
    report = passing_cpu_report()
    assert preflight.validate_child_report(report) == report
    schema = json.loads(
        (SPEC / "contracts" / "schemas" / "preflight-child-report.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert_valid(Draft202012Validator(schema), report)
    changed = dict(report)
    changed["unexpected"] = True
    with pytest.raises(ValueError, match="closed object"):
        preflight.validate_child_report(changed)


@pytest.mark.parametrize(
    ("query", "report", "expected"),
    [
        (
            preflight.GpuQueryResult(
                None,
                None,
                "CUDA_DEVICE_NOT_FOUND",
                preflight.ProcessResult(returncode=9),
            ),
            cuda_report(),
            "CUDA_DEVICE_NOT_FOUND",
        ),
        (gpu_result("GPU, 12.0, 572.60, 12227"), cuda_report(), "CUDA_DRIVER_TOO_OLD"),
        (
            gpu_result(),
            cuda_report(
                device_available=False,
                device=None,
                result="failed",
                reason_code="CUDA_DEVICE_NOT_FOUND",
            ),
            "CUDA_DEVICE_NOT_FOUND",
        ),
        (
            gpu_result(),
            cuda_report(device=cuda_device(compute_capability="7.0", arch_list=["sm_70"])),
            "CUDA_CAPABILITY_UNSUPPORTED",
        ),
        (
            gpu_result(),
            cuda_report(device=cuda_device(arch_list=["sm_50", "sm_90"])),
            "CUDA_ARCH_NOT_IN_BUILD",
        ),
        (
            gpu_result(),
            cuda_report(device=cuda_device(total_memory_mib=3583)),
            "CUDA_MEMORY_INSUFFICIENT",
        ),
        (
            gpu_result(),
            cuda_report(device=cuda_device(free_memory_mib=2047)),
            "CUDA_MEMORY_INSUFFICIENT",
        ),
        (
            gpu_result(),
            cuda_report(device=cuda_device(max_relative_difference=1.1e-5)),
            "CUDA_NUMERIC_CHECK_FAILED",
        ),
    ],
)
def test_cuda_seven_step_failure_order(
    query: preflight.GpuQueryResult,
    report: dict[str, object],
    expected: str,
) -> None:
    result = preflight.evaluate_cuda_preflight(query, report)
    assert result["result"] == "failed"
    assert result["reason_code"] == expected


@pytest.mark.parametrize(
    "device",
    [
        cuda_device(compute_capability="7.5", arch_list=["sm_75"], total_memory_mib=3584, free_memory_mib=2048, max_relative_difference=1e-5),
        cuda_device(compute_capability="8.9", arch_list=["sm_86"], total_memory_mib=3584, free_memory_mib=2048, max_relative_difference=1e-5),
        cuda_device(compute_capability="12.0", arch_list=["sm_120"]),
    ],
)
def test_cuda_boundaries_pass(device: dict[str, object]) -> None:
    result = preflight.evaluate_cuda_preflight(gpu_result(), cuda_report(device=device))
    assert (result["result"], result["reason_code"]) == ("passed", None)


def test_cuda_first_failure_is_driver_before_capability() -> None:
    result = preflight.evaluate_cuda_preflight(
        gpu_result("GPU, 7.0, 572.60, 12227"),
        cuda_report(device=cuda_device(compute_capability="7.0", arch_list=["sm_70"])),
    )
    assert result["reason_code"] == "CUDA_DRIVER_TOO_OLD"


def test_cpu_startup_is_allowed_with_missing_backend_for_reader() -> None:
    missing = passing_cpu_report()
    missing["backend_status"] = "missing"
    missing["device_available"] = False
    chosen = selection(backend_reason="BACKEND_MISSING")
    absent_query = preflight.GpuQueryResult(
        None,
        None,
        "CUDA_DEVICE_NOT_FOUND",
        preflight.ProcessResult(returncode=None, launch_error=True),
    )
    assert (
        preflight.require_startup_allowed(
            chosen, child_execution(missing), absent_query
        )
        is None
    )


def test_cuda_startup_requires_pass_and_includes_cpu_command() -> None:
    chosen = selection("wsl-cuda")
    failed_child = child_execution(
        cuda_report(device=cuda_device(free_memory_mib=2047))
    )
    with pytest.raises(preflight.ProfileSelectionError) as raised:
        preflight.require_startup_allowed(
            chosen,
            failed_child,
            gpu_result(),
            storage_root_display="~/.llm-foundations",
        )
    assert raised.value.reason_code == "CUDA_MEMORY_INSUFFICIENT"
    assert raised.value.cpu_command == (
        "envs/wsl-cpu/bin/python -m llm_foundations_companion serve --storage "
        "~/.llm-foundations"
    )
    assert raised.value.cpu_command in raised.value.message


def test_cuda_startup_pass_returns_bound_cuda_metadata() -> None:
    result = preflight.require_startup_allowed(
        selection("wsl-cuda"),
        child_execution(cuda_report()),
        gpu_result(),
    )
    assert result is not None
    assert result["result"] == "passed"
    assert result["reason_code"] is None


def test_operation_capabilities_are_closed_slice_specific_and_enforced() -> None:
    assert tuple(preflight.OPERATION_CAPABILITIES) == (
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
    capabilities = preflight.capabilities_from_preflight(
        selection(), child_execution()
    )
    assert {name for name, record in capabilities.items() if record["available"]} == {
        "tokenizer_train", "tiny_train", "tiny_resume", "evaluate"
    }
    for name, record in capabilities.items():
        if name in preflight.S2_OPERATION_NAMES:
            assert record == {"available": True}
        else:
            assert record["reason_code"] == "CAPABILITY_UNAVAILABLE"
    assert preflight.capability_enabled(capabilities, "tiny_train") is True
    preflight.require_capability(capabilities, "tiny_train")
    with pytest.raises(preflight.CapabilityUnavailable) as raised:
        preflight.require_capability(capabilities, "model_prepare")
    assert raised.value.operation == "model_prepare"


@pytest.mark.parametrize(
    ("backend_reason", "report_status", "expected"),
    [
        ("BACKEND_MISSING", "passed", "BACKEND_MISSING"),
        (None, "version_mismatch", "BACKEND_VERSION_MISMATCH"),
    ],
)
def test_missing_and_wrong_pins_disable_every_capability_with_exact_reason(
    backend_reason: str | None, report_status: str, expected: str
) -> None:
    report = passing_cpu_report()
    report["backend_status"] = report_status
    capabilities = preflight.capabilities_from_preflight(
        selection(backend_reason=backend_reason), child_execution(report)
    )
    assert {record["reason_code"] for record in capabilities.values()} == {expected}


def test_receipt_has_six_tests_schema_digest_and_no_internal_interpreter_path() -> None:
    receipt = preflight.build_preflight_receipt(
        selection(),
        child_execution(),
        storage_probe(),
        storage_root_display="~/.llm-foundations",
        started_at="2026-10-01T12:00:00.000Z",
        finished_at="2026-10-01T12:00:01.000Z",
        python_version="3.12.3",
    )
    assert receipt["status"] == "pass"
    assert len(receipt["tests"]) == 6
    assert sys.executable not in json.dumps(receipt)
    assert_valid(openapi_validator("PreflightReceipt"), receipt)
    digest = receipt["receipt_sha256"]
    receipt_with_artifact = preflight.build_preflight_receipt(
        selection(),
        child_execution(),
        storage_probe(),
        storage_root_display="~/.llm-foundations",
        started_at="2026-10-01T12:00:00.000Z",
        finished_at="2026-10-01T12:00:01.000Z",
        python_version="3.12.3",
        receipt_artifact_id="123e4567-e89b-42d3-a456-426614174000",
    )
    assert receipt_with_artifact["receipt_sha256"] == digest


def test_cuda_receipt_records_profile_failure_without_cpu_fallback() -> None:
    receipt = preflight.build_preflight_receipt(
        selection("wsl-cuda"),
        child_execution(cuda_report(device=cuda_device(free_memory_mib=2047))),
        storage_probe(),
        storage_root_display="~/.llm-foundations",
        started_at="2026-10-01T12:00:00.000Z",
        finished_at="2026-10-01T12:00:01.000Z",
        gpu_query=gpu_result(),
        python_version="3.12.3",
    )
    assert receipt["device"] == "cuda"
    assert receipt["status"] == "fail"
    assert receipt["cuda"]["reason_code"] == "CUDA_MEMORY_INSUFFICIENT"
    assert receipt["tests"][1] == {
        "id": "P00-RUN-002-PROFILE",
        "status": "fail",
        "observed": "CUDA_MEMORY_INSUFFICIENT",
    }
    assert_valid(openapi_validator("PreflightReceipt"), receipt)


def test_preflight_main_invalid_profile_emits_closed_report_without_backend_import() -> None:
    report = preflight_main.build_child_report("invalid")
    assert report["backend_status"] == "version_mismatch"
    assert set(report["versions"].values()) == {None}
    preflight.validate_child_report(report)
