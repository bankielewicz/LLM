from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest


VALIDATOR_PATH = Path(__file__).resolve().parents[1] / "locks" / "validate.py"
SPEC = importlib.util.spec_from_file_location("dependency_lock_validator", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)

GENERATOR_PATH = (
    Path(__file__).resolve().parents[1] / "locks" / "generate_wheel_manifests.py"
)
GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "dependency_lock_generator", GENERATOR_PATH
)
assert GENERATOR_SPEC is not None and GENERATOR_SPEC.loader is not None
generator = importlib.util.module_from_spec(GENERATOR_SPEC)
sys.modules[GENERATOR_SPEC.name] = generator
try:
    GENERATOR_SPEC.loader.exec_module(generator)
finally:
    del sys.modules[GENERATOR_SPEC.name]

MATERIALIZER_PATH = (
    Path(__file__).resolve().parents[1] / "locks" / "materialize_wheelhouse.py"
)
MATERIALIZER_SPEC = importlib.util.spec_from_file_location(
    "dependency_lock_materializer", MATERIALIZER_PATH
)
assert MATERIALIZER_SPEC is not None and MATERIALIZER_SPEC.loader is not None
PREVIOUS_VALIDATE = sys.modules.get("validate")
sys.modules["validate"] = validator
try:
    materializer = importlib.util.module_from_spec(MATERIALIZER_SPEC)
    MATERIALIZER_SPEC.loader.exec_module(materializer)
finally:
    if PREVIOUS_VALIDATE is None:
        del sys.modules["validate"]
    else:
        sys.modules["validate"] = PREVIOUS_VALIDATE


def test_complete_dependency_lock_set_is_internally_consistent() -> None:
    summary = validator.validate()
    assert summary["status"] == "PASS"
    assert {
        profile: result["package_count"]
        for profile, result in summary["profiles"].items()
    } == {
        "win-cpu": 39,
        "win-cuda": 39,
        "wsl-cpu": 38,
        "wsl-cuda": 53,
    }
    assert summary["legacy_test"]["package_count"] == 10


def test_lock_parser_rejects_a_package_without_hashes(tmp_path: Path) -> None:
    broken = tmp_path / "broken.requirements.txt"
    broken.write_text(
        "example==1.0 \\\n    # from https://pypi.org/simple\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(validator.LockValidationError, match="has no SHA-256"):
        validator.parse_lock(broken)


def test_lock_parser_rejects_ambiguous_source_indexes(tmp_path: Path) -> None:
    broken = tmp_path / "broken.requirements.txt"
    broken.write_text(
        "example==1.0 \\\n"
        "    --hash=sha256:" + "0" * 64 + "\n"
        "    # from https://pypi.org/simple\n"
        "    # from https://example.invalid/simple\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(validator.LockValidationError, match="one source index"):
        validator.parse_lock(broken)


def test_wheelhouse_validator_rejects_an_incomplete_directory(tmp_path: Path) -> None:
    with pytest.raises(validator.LockValidationError, match="file set differs"):
        validator.validate_wheelhouse("legacy-test", tmp_path)


@pytest.mark.parametrize(
    "parser",
    [
        pytest.param(validator.parse_lock, id="offline-validator"),
        pytest.param(generator.parse_lock, id="manifest-generator"),
    ],
)
@pytest.mark.parametrize(
    "unexpected",
    [
        "evil>=2",
        "evil @ https://example.invalid/evil.whl",
        "-r injected.requirements.txt",
        "--extra-index-url https://example.invalid/simple",
    ],
)
def test_lock_parsers_reject_every_unknown_active_line(
    tmp_path: Path, parser: object, unexpected: str
) -> None:
    broken = tmp_path / "broken.requirements.txt"
    broken.write_text(
        "example==1.0 \\\n"
        "    --hash=sha256:" + "0" * 64 + "\n"
        "    # from https://pypi.org/simple\n"
        f"{unexpected}\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="unsupported active lock line"):
        parser(broken)  # type: ignore[operator]


def test_profile_rejects_replacement_of_a_d10_direct_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = "win-cpu"
    lock_name = f"{profile}.requirements.txt"
    manifest_name = f"{profile}.wheels.json"
    original_locks = validator.LOCKS
    changed_lock = (
        (original_locks / lock_name)
        .read_text(encoding="utf-8")
        .replace("accelerate==1.10.1", "fakeproject==1.10.1", 1)
    )
    (tmp_path / lock_name).write_text(changed_lock, encoding="utf-8", newline="\n")
    changed_digest = hashlib.sha256((tmp_path / lock_name).read_bytes()).hexdigest()
    manifest = json.loads((original_locks / manifest_name).read_text(encoding="utf-8"))
    manifest["lock_sha256"] = changed_digest
    manifest["packages"][0]["name"] = "fakeproject"
    (tmp_path / manifest_name).write_text(
        json.dumps(manifest), encoding="utf-8", newline="\n"
    )
    expected = dict(validator.PROFILES[profile])
    expected["lock_sha256"] = changed_digest
    monkeypatch.setattr(validator, "LOCKS", tmp_path)

    with pytest.raises(
        validator.LockValidationError, match="D10 direct pin changed: accelerate"
    ):
        validator.validate_profile(profile, expected)


def test_materializer_retains_partial_bytes_and_failure_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    locks = tmp_path / "locks"
    locks.mkdir()
    expected_bytes = b"good"
    received_bytes = b"bad!"
    filename = "example-1.0-py3-none-any.whl"
    manifest = {
        "packages": [
            {
                "name": "example",
                "version": "1.0",
                "filename": filename,
                "source_index": "https://example.invalid/simple",
                "url": f"https://example.invalid/files/{filename}",
                "bytes": len(expected_bytes),
                "sha256": hashlib.sha256(expected_bytes).hexdigest(),
            }
        ]
    }
    (locks / "legacy-test.wheels.json").write_text(
        json.dumps(manifest), encoding="utf-8", newline="\n"
    )
    monkeypatch.setattr(materializer.lock_validator, "LOCKS", locks)
    monkeypatch.setattr(materializer.lock_validator, "validate", lambda: None)
    monkeypatch.setattr(
        materializer.urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(received_bytes),
    )

    destination = tmp_path / "wheelhouse"
    with pytest.raises(validator.LockValidationError, match="SHA-256"):
        materializer.materialize("legacy-test", destination)

    partial_path = destination / f"{filename}.part"
    assert partial_path.read_bytes() == received_bytes
    assert not (destination / filename).exists()
    receipt = json.loads(
        (destination / materializer.FAILURE_RECEIPT).read_text(encoding="utf-8")
    )
    assert receipt["status"] == "FAIL"
    assert receipt["expected"] == {
        "bytes": len(expected_bytes),
        "sha256": hashlib.sha256(expected_bytes).hexdigest(),
    }
    assert receipt["actual"] == {
        "bytes": len(received_bytes),
        "sha256": hashlib.sha256(received_bytes).hexdigest(),
        "partial_file": partial_path.name,
    }
    assert receipt["reason"]["type"] == "LockValidationError"
