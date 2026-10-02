"""Run the unchanged foundations lab suite and retain an exact denominator."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import unittest


EXPECTED = (
    "test_lab.LabTests.test_evaluation_leaves_weights_and_training_mode_unchanged",
    "test_lab.LabTests.test_future_input_cannot_change_earlier_predictions",
    "test_lab.LabTests.test_one_valid_window_and_shifted_targets",
    "test_lab.LabTests.test_serialized_tokenizer_retains_exact_ids",
    "test_lab.LabTests.test_split_overlap_rejected",
    "test_lab.LabTests.test_training_resume_generation_and_provenance",
    "test_lab.LabTests.test_whitespace_unicode_and_unseen_bytes_round_trip",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_snapshot(labs: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(labs.rglob("*")):
        if (
            not path.is_file()
            or "__pycache__" in path.parts
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        raw = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(labs).as_posix(),
                "size_bytes": len(raw),
                "sha256": sha256(raw),
            }
        )
    return rows


class RetainedResult(unittest.TextTestResult):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.rows: dict[str, dict[str, object]] = {}
        self.started: dict[str, float] = {}

    def startTest(self, test: unittest.case.TestCase) -> None:
        self.started[test.id()] = time.monotonic()
        super().startTest(test)

    def _record(
        self, test: unittest.case.TestCase, status: str, reason: str | None = None
    ) -> None:
        row: dict[str, object] = {
            "test": test.id(),
            "status": status,
            "seconds": round(time.monotonic() - self.started.get(test.id(), 0.0), 3),
        }
        if reason:
            row["reason"] = reason[:4_000]
        self.rows[test.id()] = row

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
        self._record(test, "BLOCKED", reason)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-course", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    course = args.canonical_course.resolve(strict=True)
    labs = (course / "labs").resolve(strict=True)
    if labs.parent != course or not (labs / "test_lab.py").is_file():
        parser.error("--canonical-course must contain labs/test_lab.py")
    reports = args.report_dir.resolve()
    reports.mkdir(parents=True, exist_ok=False)

    before = source_snapshot(labs)
    raw_output = io.StringIO()
    result: RetainedResult | None = None
    harness_error: str | None = None
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(labs))
    try:
        suite = unittest.defaultTestLoader.discover(
            str(labs), pattern="test_lab.py", top_level_dir=str(labs)
        )
        runner = unittest.TextTestRunner(
            stream=raw_output, verbosity=2, resultclass=RetainedResult
        )
        result = runner.run(suite)
    except BaseException:
        harness_error = traceback.format_exc(limit=20)
    finally:
        try:
            sys.path.remove(str(labs))
        except ValueError:
            pass

    after = source_snapshot(labs)
    unchanged = before == after
    observed = result.rows if result is not None else {}
    rows = []
    for identifier in EXPECTED:
        row = observed.get(identifier)
        if row is None:
            row = {
                "test": identifier,
                "status": "NOT_RUN",
                "reason": "The exact protected legacy test did not execute.",
            }
        rows.append(row)
    extras = sorted(set(observed) - set(EXPECTED))
    denominator_ok = not extras and len(observed) == len(EXPECTED)
    passed = (
        result is not None
        and result.wasSuccessful()
        and denominator_ok
        and unchanged
        and all(row["status"] == "PASS" for row in rows)
    )

    raw = raw_output.getvalue()
    if harness_error:
        raw += "\nHARNESS ERROR\n" + harness_error
    (reports / "unittest.txt").write_text(raw, encoding="utf-8")
    import torch

    payload = {
        "format": "llm-foundations-s2-legacy-checks-v1",
        "status": "PASS" if passed else "FAIL",
        "s2_native_case": {
            "case_id": "S2-NATIVE-024",
            "status": "PASS" if passed else "FAIL",
            "observation": {
                "legacy_test_total": len(EXPECTED),
                "legacy_test_passed": sum(row["status"] == "PASS" for row in rows),
                "protected_lab_bytes_unchanged": unchanged,
            },
        },
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "tests": rows,
        "unexpected_tests": extras,
        "source_before": before,
        "source_after": after,
        "harness_error": harness_error,
        "product_qualification": "NOT_RUN",
    }
    (reports / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for row in rows:
        print(f"{row['test']}: {row['status']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
