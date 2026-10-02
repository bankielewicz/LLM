"""Run the frozen S2 installed-model development denominator.

This runner is executed by the locked runtime interpreter, never by the
Torch-free authoring environment.  It imports the installed wheel while test
definitions remain outside the wheel under scripts/native_s2.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import email.parser
import hashlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback
import unittest
import zipfile


def case(
    number: int, name: str, requirements: tuple[str, ...], expected: str
) -> tuple[str, dict[str, object]]:
    identifier = f"S2-NATIVE-{number:03d}"
    return identifier, {
        "case_id": identifier,
        "name": name,
        "requirements": list(requirements),
        "expected_reference": expected,
    }


CASES = dict(
    [
        case(1, "Installed candidate and locked runtime identity", ("DEL-001", "DEL-002"), "Installed candidate wheel; CPython 3.12; locked WSL CPU Torch 2.8.0+cpu."),
        case(2, "Fixed S2 dispatch and closed future operations", ("RUN-002", "DEL-001"), "Exactly six S2 handlers are fixed and available; all nine later operations fail closed."),
        case(3, "Standalone byte tokenizer", ("INT-003",), "A real byte-257 tokenizer job reports zero merges, byte type and the protected fingerprint."),
        case(4, "Deterministic BPE tokenizer", ("INT-003",), "Two BPE-320 executions preserve record order and produce identical IDs, merges, bytes and fingerprint."),
        case(5, "Tokenizer and training text byte cap", ("INT-003", "INT-007"), "A 200001-byte train split is rejected before queue mutation while validation bytes are not added to that cap."),
        case(6, "Real 50-update tiny training", ("INT-003", "INT-004"), "Real Torch training completes 50 updates with finite metrics and durable checkpoints at 0, 25 and 50."),
        case(7, "Safe checkpoint format and identities", ("INT-004",), "The seven-file checkpoint uses safetensors/JSON only and every manifest, schema, size and digest check passes."),
        case(8, "Uninterrupted 50 versus 25 plus 25 resume", ("INT-004",), "Tensors, optimizer state, batch RNG, fixed validation values and greedy IDs match exactly on WSL CPU."),
        case(9, "Byte-normalized evaluation arithmetic", ("INT-005",), "Aggregate arithmetic is 10/11; the uniform one-byte fixture is 2 ln 257; every target including EOS is scored."),
        case(10, "Real evaluation leaves weights untouched", ("INT-003", "INT-005"), "Full-split evaluation creates no checkpoint and leaves every checkpoint/model byte unchanged."),
        case(11, "Context preview performs tokenizer work only", ("INT-006",), "A real preview succeeds without loading or changing learned weights and commits provenance only."),
        case(12, "Matching-preview tiny generation", ("INT-006",), "Generation uses the matching preview, produces specified IDs/count/reason and leaves weights unchanged."),
        case(13, "Canonical numeric preview spelling", ("INT-006", "INT-008"), "Numerically equal 1 and 1.0 produce the same canonical digest and generation is accepted."),
        case(14, "Stale preview prequeue rejection matrix", ("INT-006", "INT-008"), "Prompt, seed, budget, subject and corrupt-preview changes reject before job or inference mutation."),
        case(15, "Accepted-worker preview revalidation", ("INT-006",), "A preview changed after admission fails CONTEXT_PREVIEW_STALE before inference."),
        case(16, "Frozen tiny decode oracle", ("INT-006",), "All tiny selection and stop rows preserve tie order, nucleus set, EOS/count and visible-text behavior."),
        case(17, "Incompatible checkpoint identity matrix", ("INT-004",), "Dataset, tokenizer, config, lock and profile mismatches reject before queueing and change no state."),
        case(18, "Tampered checkpoint matrix", ("INT-004",), "Bad digest/name/shape/dtype, missing optimizer and alias cycle reject before effective tensor use."),
        case(19, "Inference-only checkpoint cannot resume", ("INT-004",), "Inference-only export has a distinct digest and is rejected for resume."),
        case(20, "Cancel the prescribed step-25 hold", ("RUN-009", "INT-004"), "The 50/eval-25 diagnosis job cancels in cancellable_hold and retains interrupted/user_cancelled step 25."),
        case(21, "Resume the held checkpoint to step 50", ("RUN-009", "INT-004"), "A new child resumes the safe step-25 checkpoint for 25 additional updates and completes at 50."),
        case(22, "Initial evaluation has no update effect", ("INT-003", "INT-004"), "Step-zero evaluation changes no weight and consumes no batch RNG."),
        case(23, "Protected pure-mechanics parity", ("INT-003", "RUN-013"), "Protected and companion mechanics match exact IDs/targets/counts/greedy output and 1e-6 logits/loss."),
        case(24, "Preserved foundations lab suite", ("RUN-013",), "All seven unchanged protected lab tests pass in the separate locked legacy environment."),
        case(25, "CLI/API parity without shell execution", ("RUN-013",), "CLI and API canonical records match; learner text never becomes shell, argv, eval, exec or a dynamic import."),
    ]
)

TEST_CASE_IDS = frozenset(CASES) - {"S2-NATIVE-001", "S2-NATIVE-002", "S2-NATIVE-024"}
CAPABILITY_BLOCKED_IDS = frozenset(
    {
        "S2-NATIVE-002",
        "S2-NATIVE-011",
        "S2-NATIVE-012",
        "S2-NATIVE-013",
        "S2-NATIVE-014",
        "S2-NATIVE-015",
        "S2-NATIVE-025",
    }
)
S2_OPERATIONS = frozenset(
    {"tokenizer_train", "tiny_train", "tiny_resume", "evaluate", "generate", "context_preview"}
)
CASE_RE = re.compile(r"(?:^|\.)test_s2_native_(?P<number>[0-9]{3})(?:_|$)")
FROZEN_LIKE_RE = re.compile(r"(?:^|\.)test_s2_native_")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def report_json_default(value: object) -> object:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not reportable")


def row(
    identifier: str,
    status: str,
    *,
    observation: object | None = None,
    reason: str | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
) -> dict[str, object]:
    output = dict(CASES[identifier])
    output.update(
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        actual_observation=observation,
    )
    if reason:
        output["reason"] = reason[:8_000]
    return output


def wheel_metadata(wheel: Path) -> dict[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        members = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(members) != 1:
            raise AssertionError("Candidate wheel has ambiguous distribution metadata")
        parsed = email.parser.BytesParser().parsebytes(archive.read(members[0]))
    if parsed["Name"] != "llm-foundations-companion" or not parsed["Version"]:
        raise AssertionError("Candidate wheel metadata identity differs")
    return {"name": parsed["Name"], "version": parsed["Version"]}


def installed_identity(wheel: Path) -> dict[str, object]:
    import torch
    import llm_foundations_companion

    candidate = wheel_metadata(wheel)
    installed_version = metadata.version("llm-foundations-companion")
    assert installed_version == candidate["version"] == llm_foundations_companion.__version__
    assert sys.version_info[:2] == (3, 12)
    assert platform.python_implementation() == "CPython"
    assert platform.system() == "Linux" and platform.machine() == "x86_64"
    assert str(torch.__version__) == "2.8.0+cpu"
    assert torch.version.cuda is None and not torch.cuda.is_available()
    return {
        "wheel_sha256": sha256(wheel.read_bytes()),
        "distribution_version": installed_version,
        "python": platform.python_version(),
        "python_executable_sha256": sha256(Path(sys.executable).read_bytes()),
        "torch": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }


def dispatch_identity() -> dict[str, object]:
    from llm_foundations_companion.operations import HANDLERS, OPERATIONS, OperationUnavailable
    from llm_foundations_companion.preflight import OPERATION_CAPABILITIES

    assert tuple(HANDLERS) == OPERATIONS and len(OPERATIONS) == 15
    available = {
        name for name, record in OPERATION_CAPABILITIES.items()
        if record.get("available") is True
    }
    assert available == S2_OPERATIONS, (available, S2_OPERATIONS)
    unavailable_handlers = set()
    for operation in OPERATIONS:
        handler = HANDLERS[operation]
        if operation in available:
            assert handler.__module__.startswith("llm_foundations_companion.")
            continue
        unavailable_handlers.add(handler)
        try:
            handler({"operation": operation}, object())
        except OperationUnavailable as exc:
            assert exc.code == "CAPABILITY_UNAVAILABLE"
        else:
            raise AssertionError(f"Future operation {operation} did not fail closed")
    assert len(unavailable_handlers) == 1
    assert all(HANDLERS[name] not in unavailable_handlers for name in available)
    return {
        "available_handlers": sorted(available),
        "unavailable_handlers": sorted(set(OPERATIONS) - available),
        "operation_count": len(OPERATIONS),
    }


class NativeResult(unittest.TextTestResult):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.case_rows: dict[str, dict[str, object]] = {}
        self.started_at: dict[str, str] = {}
        self.started_clock: dict[str, float] = {}
        self.unknown_tests: list[str] = []
        self.support_rows: list[dict[str, object]] = []

    @staticmethod
    def case_id(test: unittest.case.TestCase) -> str | None:
        match = CASE_RE.search(test.id())
        return f"S2-NATIVE-{match.group('number')}" if match else None

    def startTest(self, test: unittest.case.TestCase) -> None:
        identifier = self.case_id(test)
        if identifier is None and FROZEN_LIKE_RE.search(test.id()):
            self.unknown_tests.append(test.id())
        elif identifier is not None and identifier not in TEST_CASE_IDS:
            self.unknown_tests.append(test.id())
        elif identifier in self.case_rows:
            self.unknown_tests.append("duplicate:" + identifier + ":" + test.id())
        self.started_at[test.id()] = now()
        self.started_clock[test.id()] = time.monotonic()
        super().startTest(test)

    def _record(
        self, test: unittest.case.TestCase, status: str, reason: str | None = None
    ) -> None:
        identifier = self.case_id(test)
        observation = getattr(test, "s2_observation", None)
        if status == "NOT_RUN":
            observation = None
        elif observation is None:
            observation = {
                "test": test.id(),
                "seconds": round(
                    time.monotonic() - self.started_clock.get(test.id(), 0.0), 3
                ),
            }
        started_at = (
            None if status == "NOT_RUN" else self.started_at.get(test.id())
        )
        ended_at = None if status == "NOT_RUN" else now()
        if identifier is None:
            self.support_rows.append(
                {
                    "test": test.id(),
                    "status": status,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "actual_observation": observation,
                    **({"reason": reason[:8_000]} if reason else {}),
                }
            )
            return
        if identifier not in TEST_CASE_IDS or identifier in self.case_rows:
            return
        self.case_rows[identifier] = row(
            identifier,
            status,
            observation=observation,
            reason=reason,
            started_at=started_at,
            ended_at=ended_at,
        )

    def addSuccess(self, test: unittest.case.TestCase) -> None:
        super().addSuccess(test)
        self._record(test, "PASS")

    def addFailure(
        self, test: unittest.case.TestCase, err: tuple[type[BaseException], BaseException, object]
    ) -> None:
        super().addFailure(test, err)
        self._record(test, "FAIL", self._exc_info_to_string(err, test))

    def addError(
        self, test: unittest.case.TestCase, err: tuple[type[BaseException], BaseException, object]
    ) -> None:
        super().addError(test, err)
        self._record(test, "FAIL", self._exc_info_to_string(err, test))

    def addSkip(self, test: unittest.case.TestCase, reason: str) -> None:
        super().addSkip(test, reason)
        status = "NOT_RUN" if reason.startswith("NOT_RUN:") else "BLOCKED"
        self._record(test, status, reason)

    def addExpectedFailure(
        self,
        test: unittest.case.TestCase,
        err: tuple[type[BaseException], BaseException, object],
    ) -> None:
        super().addExpectedFailure(test, err)
        self._record(test, "BLOCKED", self._exc_info_to_string(err, test))

    def addUnexpectedSuccess(self, test: unittest.case.TestCase) -> None:
        super().addUnexpectedSuccess(test)
        self._record(test, "FAIL", "Unexpected success")


def execute_builtin(identifier: str, action: object) -> dict[str, object]:
    started = now()
    try:
        observation = action()
        return row(identifier, "PASS", observation=observation, started_at=started, ended_at=now())
    except BaseException:
        return row(
            identifier,
            "FAIL",
            reason=traceback.format_exc(limit=20),
            started_at=started,
            ended_at=now(),
        )


def without_cases(
    suite: unittest.TestSuite,
    excluded: frozenset[str],
) -> unittest.TestSuite:
    """Return a suite without explicitly blocked frozen cases."""

    retained = unittest.TestSuite()
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            nested = without_cases(item, excluded)
            if nested.countTestCases():
                retained.addTest(nested)
            continue
        identifier = NativeResult.case_id(item)
        if identifier not in excluded:
            retained.addTest(item)
    return retained


def legacy_row(path: Path | None) -> dict[str, object]:
    if path is None:
        return row(
            "S2-NATIVE-024",
            "NOT_RUN",
            reason="No independently executed legacy result was supplied.",
        )
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
        legacy = payload["s2_native_case"]
        assert payload["format"] == "llm-foundations-s2-legacy-checks-v1"
        assert legacy["case_id"] == "S2-NATIVE-024"
        assert payload["status"] == "PASS" and legacy["status"] == "PASS"
        return row(
            "S2-NATIVE-024",
            "PASS",
            observation={
                **legacy["observation"],
                "legacy_result_path": str(path),
                "legacy_result_sha256": sha256(raw),
            },
        )
    except BaseException:
        return row(
            "S2-NATIVE-024",
            "FAIL",
            reason=traceback.format_exc(limit=20),
        )


def artifact_manifest(root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "result.json":
            continue
        raw = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": len(raw),
                "sha256": sha256(raw),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--legacy-result", type=Path)
    parser.add_argument(
        "--exclude-capability-blocked",
        action="store_true",
        help=(
            "retain pending APP-009 cases as BLOCKED while running all independent "
            "installed checks; this mode can never pass the development gate"
        ),
    )
    parser.add_argument("--list-plan", action="store_true")
    args = parser.parse_args()

    if args.list_plan:
        print(json.dumps({"denominator": len(CASES), "cases": list(CASES.values())}, indent=2))
        return 0
    if args.report_dir is None or args.wheel is None:
        parser.error("--report-dir and --wheel are required unless --list-plan is used")

    repo = Path(__file__).resolve().parents[1]
    test_root = repo / "scripts" / "native_s2"
    wheel = args.wheel.resolve(strict=True)
    reports = args.report_dir.resolve()
    reports.mkdir(parents=True, exist_ok=False)
    artifacts = reports / "native-artifacts"
    artifacts.mkdir()
    os.environ["LLMF_S2_NATIVE_ARTIFACT_ROOT"] = str(artifacts)
    os.environ["LLMF_S2_CANDIDATE_WHEEL"] = str(wheel)

    excluded = CAPABILITY_BLOCKED_IDS if args.exclude_capability_blocked else frozenset()
    exclusion_reason = (
        "Pending APP-009 keeps context-preview and generation capabilities disabled; "
        "this case was retained in the denominator but intentionally not executed."
    )
    rows: dict[str, dict[str, object]] = {
        identifier: row(identifier, "BLOCKED", reason=exclusion_reason)
        for identifier in excluded
    }
    rows["S2-NATIVE-001"] = execute_builtin(
        "S2-NATIVE-001", lambda: installed_identity(wheel)
    )
    if "S2-NATIVE-002" not in excluded:
        rows["S2-NATIVE-002"] = execute_builtin(
            "S2-NATIVE-002", dispatch_identity
        )

    raw_output = io.StringIO()
    native_result: NativeResult | None = None
    discovery_error: str | None = None
    if test_root.is_dir():
        try:
            suite = unittest.defaultTestLoader.discover(
                str(test_root), pattern="test_*.py", top_level_dir=str(test_root)
            )
            suite = without_cases(suite, excluded)
            runner = unittest.TextTestRunner(
                stream=raw_output, verbosity=2, resultclass=NativeResult
            )
            native_result = runner.run(suite)
            rows.update(native_result.case_rows)
        except BaseException:
            discovery_error = traceback.format_exc(limit=20)
    else:
        discovery_error = f"Native test directory is absent: {test_root}"
    if discovery_error:
        raw_output.write("\nDISCOVERY ERROR\n" + discovery_error + "\n")
    (reports / "unittest.txt").write_text(raw_output.getvalue(), encoding="utf-8")

    rows["S2-NATIVE-024"] = legacy_row(
        args.legacy_result.resolve(strict=True) if args.legacy_result else None
    )
    for identifier in CASES:
        if identifier not in rows:
            rows[identifier] = row(
                identifier,
                "NOT_RUN",
                reason="The required native development case was not discovered or executed.",
            )

    ordered = [rows[identifier] for identifier in CASES]
    unknown = native_result.unknown_tests if native_result is not None else []
    support_rows = native_result.support_rows if native_result is not None else []
    passed = (
        not excluded
        and len(ordered) == 25
        and discovery_error is None
        and not unknown
        and all(item["status"] == "PASS" for item in support_rows)
        and native_result is not None
        and native_result.wasSuccessful()
        and all(item["status"] == "PASS" for item in ordered)
    )
    counts = {
        status: sum(item["status"] == status for item in ordered)
        for status in ("PASS", "FAIL", "BLOCKED", "NOT_RUN")
    }
    payload = {
        "format": "llm-foundations-s2-native-development-v1",
        "s2_native_development_gate": "PASS" if passed else "FAIL",
        "profile": "wsl-cpu",
        "denominator": 25,
        "counts": counts,
        "cases": ordered,
        "supporting_tests": support_rows,
        "unknown_or_duplicate_tests": unknown,
        "discovery_error": discovery_error,
        "development_exclusions": {
            "enabled": bool(excluded),
            "case_ids": sorted(excluded),
            "reason": exclusion_reason if excluded else None,
        },
        "artifacts": artifact_manifest(reports),
        "canonical_acceptance": {
            "execution_units": 605,
            "passed": 0,
            "not_run": 605,
            "product_qualification": "NOT_RUN",
        },
        "mapped_s2_acceptance": {
            "case_ids": [
                "AC-INT-003", "AC-INT-004", "AC-INT-005", "AC-INT-006", "AC-RUN-013"
            ],
            "execution_units": 20,
            "passed": 0,
            "not_run": 20,
        },
        "native_profile_qualification": {
            name: "NOT_RUN"
            for name in ("win-cpu", "win-cuda", "wsl-cpu", "wsl-cuda")
        },
        "scope": (
            "Installed-candidate WSL CPU development evidence. Component tests do "
            "not replace service integration, and this report establishes no "
            "browser, product, release, model-quality or profile qualification."
        ),
    }
    (reports / "result.json").write_text(
        json.dumps(
            payload, indent=2, sort_keys=True, default=report_json_default
        )
        + "\n",
        encoding="utf-8",
    )
    for item in ordered:
        print(f"{item['case_id']}: {item['status']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
