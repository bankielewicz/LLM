"""Fixed backend-import child for the S1 startup preflight."""

from __future__ import annotations

import json
import math
import os
import sys
from types import ModuleType
from typing import Any, Mapping


FORMAT = "llm-foundations-preflight-child-v1"
PROFILES = frozenset({"win-cpu", "win-cuda", "wsl-cpu", "wsl-cuda"})
COMMON_PINS = {
    "transformers": "4.57.1",
    "peft": "0.17.1",
    "accelerate": "1.10.1",
    "safetensors": "0.6.2",
}


def _empty_versions() -> dict[str, str | None]:
    return {
        "torch": None,
        "transformers": None,
        "peft": None,
        "accelerate": None,
        "safetensors": None,
    }


def _load_backend(name: str) -> ModuleType:
    if name == "torch":
        import torch

        return torch
    if name == "transformers":
        import transformers

        return transformers
    if name == "peft":
        import peft

        return peft
    if name == "accelerate":
        import accelerate

        return accelerate
    if name == "safetensors":
        import safetensors

        return safetensors
    raise ValueError("backend name is not fixed")


def _failed_cuda(
    reason: str, device: Mapping[str, object] | None = None
) -> dict[str, object]:
    return {"result": "failed", "reason_code": reason, "device": device}


def _report(
    *,
    backend_status: str,
    device_available: bool,
    versions: Mapping[str, str | None],
    error_text: str,
    cuda: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "format": FORMAT,
        "backend_status": backend_status,
        "device_available": device_available,
        "versions": dict(versions),
        "error_text": error_text[:2_000],
        "cuda": dict(cuda) if cuda is not None else None,
    }


def _version(module: ModuleType) -> str | None:
    value = getattr(module, "__version__", None)
    return str(value)[:80] if value is not None else None


def _arch_supports(arch_list: list[str], major: int, minor: int) -> bool:
    for entry in arch_list:
        if not entry.startswith("sm_"):
            continue
        digits = "".join(character for character in entry[3:] if character.isdigit())
        if len(digits) not in {2, 3}:
            continue
        if int(digits[:-1]) == major and int(digits[-1]) <= minor:
            return True
    return False


def _cuda_report(torch: Any) -> tuple[bool, dict[str, object], str]:
    if not bool(torch.cuda.is_available()) or int(torch.cuda.device_count()) <= 0:
        return (
            False,
            _failed_cuda("CUDA_DEVICE_NOT_FOUND"),
            "Torch found no CUDA device.",
        )
    try:
        count = int(torch.cuda.device_count())
        name = str(torch.cuda.get_device_name(0))[:200]
        major, minor = (int(value) for value in torch.cuda.get_device_capability(0))
        arch_list = [str(item) for item in torch.cuda.get_arch_list()][:32]
        properties = torch.cuda.get_device_properties(0)
        total_memory_mib = int(properties.total_memory) // (1024 * 1024)
        free_bytes, _ = torch.cuda.mem_get_info(0)
        free_memory_mib = int(free_bytes) // (1024 * 1024)
    except Exception:
        return (
            False,
            _failed_cuda("CUDA_DEVICE_NOT_FOUND"),
            "CUDA device metadata could not be read.",
        )

    device: dict[str, object] = {
        "name": name,
        "count": count,
        "compute_capability": f"{major}.{minor}",
        "arch_list": arch_list,
        "total_memory_mib": total_memory_mib,
        "free_memory_mib": free_memory_mib,
        "max_relative_difference": None,
    }
    if (major, minor) < (7, 5):
        return (
            False,
            _failed_cuda("CUDA_CAPABILITY_UNSUPPORTED", device),
            "CUDA compute capability is below 7.5.",
        )
    if not _arch_supports(arch_list, major, minor):
        return (
            False,
            _failed_cuda("CUDA_ARCH_NOT_IN_BUILD", device),
            "The Torch build has no compatible compiled CUDA architecture.",
        )
    if total_memory_mib < 3_584 or free_memory_mib < 2_048:
        return (
            False,
            _failed_cuda("CUDA_MEMORY_INSUFFICIENT", device),
            "CUDA memory is below the startup threshold.",
        )

    old_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        values = (
            torch.arange(64 * 64, dtype=torch.float32)
            .remainder(97)
            .div(97)
            .reshape(64, 64)
        )
        cpu_result = values @ values.transpose(0, 1)
        cuda_values = values.to("cuda:0")
        cuda_result = (cuda_values @ cuda_values.transpose(0, 1)).cpu()
        numerator = float((cpu_result - cuda_result).abs().max().item())
        denominator = float(cpu_result.abs().max().item())
        difference = numerator / denominator
    except Exception:
        return (
            False,
            _failed_cuda("CUDA_NUMERIC_CHECK_FAILED", device),
            "The fixed CUDA numeric check could not complete.",
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    if not math.isfinite(difference) or difference > 1e-5:
        device["max_relative_difference"] = (
            difference if math.isfinite(difference) else None
        )
        return (
            False,
            _failed_cuda("CUDA_NUMERIC_CHECK_FAILED", device),
            "The fixed CUDA numeric check exceeded 1e-5.",
        )
    device["max_relative_difference"] = difference
    return (
        True,
        {"result": "passed", "reason_code": None, "device": device},
        "",
    )


def build_child_report(profile: str | None = None) -> dict[str, object]:
    profile = profile or os.environ.get("LLM_FOUNDATIONS_PROFILE")
    versions = _empty_versions()
    if profile not in PROFILES:
        return _report(
            backend_status="version_mismatch",
            device_available=False,
            versions=versions,
            error_text="The selected runtime profile is invalid.",
            cuda=None,
        )
    modules: dict[str, ModuleType] = {}
    missing: list[str] = []
    for name in versions:
        try:
            module = _load_backend(name)
        except Exception:
            missing.append(name)
            continue
        modules[name] = module
        versions[name] = _version(module)
    if missing:
        return _report(
            backend_status="missing",
            device_available=False,
            versions=versions,
            error_text=(
                "Required backend imports unavailable: "
                + ", ".join(sorted(missing))
                + "."
            ),
            cuda=(
                _failed_cuda("CUDA_DEVICE_NOT_FOUND")
                if profile.endswith("-cuda")
                else None
            ),
        )

    expected = dict(COMMON_PINS)
    expected["torch"] = (
        "2.8.0+cu128" if profile.endswith("-cuda") else "2.8.0+cpu"
    )
    mismatched = sorted(
        name for name, value in versions.items() if value != expected[name]
    )
    if mismatched:
        return _report(
            backend_status="version_mismatch",
            device_available=False,
            versions=versions,
            error_text="Locked backend version mismatch: " + ", ".join(mismatched) + ".",
            cuda=(
                _failed_cuda("CUDA_DEVICE_NOT_FOUND")
                if profile.endswith("-cuda")
                else None
            ),
        )

    torch = modules["torch"]
    if profile.endswith("-cuda"):
        available, cuda, error_text = _cuda_report(torch)
        return _report(
            backend_status="passed",
            device_available=available,
            versions=versions,
            error_text=error_text,
            cuda=cuda,
        )
    try:
        probe = torch.empty((1,), dtype=torch.float32, device="cpu")
        probe.zero_()
    except Exception:
        return _report(
            backend_status="passed",
            device_available=False,
            versions=versions,
            error_text="CPU device allocation failed.",
            cuda=None,
        )
    return _report(
        backend_status="passed",
        device_available=True,
        versions=versions,
        error_text="",
        cuda=None,
    )


def main() -> int:
    try:
        report = build_child_report()
        output = json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except Exception:
        output = json.dumps(
            _report(
                backend_status="missing",
                device_available=False,
                versions=_empty_versions(),
                error_text="The bounded backend preflight child failed.",
                cuda=None,
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
    sys.stdout.write(output + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
