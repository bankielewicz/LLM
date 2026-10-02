from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import pytest
from coverage import CoverageData

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_s2_gate as gate_module


COMMIT = "a" * 40
TREE = "b" * 40
LABS = [
    f"static/course/labs/{name}.py"
    for name in ("data", "lab", "make_demo_data", "model", "test_lab", "tokenizer")
]
SCOPE = (
    "development coverage for the exact bound Python package inventory; "
    "ordinary locked-runtime and protected-legacy qualification remain separate"
)


def _raw(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + chr(10)
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _fixture(
    tmp_path: Path,
    *,
    under_threshold: bool = False,
    missing_files: int = 0,
    exclusions: bool = False,
    invalid_raw: bool = False,
    native_ghost: bool = False,
    legacy_missing_child_entrypoint: bool = False,
    extra_phase_raw: bool = False,
    fabricated_report_lines: bool = False,
    wrong_executed_root: bool = False,
    unrelated_combined: bool = False,
    pid_mismatch: bool = False,
    copied_native_raw: bool = False,
) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    package = repo / "companion/src/llm_foundations_companion"
    package.mkdir(parents=True)
    relative_names = sorted(
        LABS
        + ["__main__.py", "service.py", "worker_main.py"]
        + [f"module_{index:02d}.py" for index in range(45)]
    )
    inventory: dict[str, dict[str, object]] = {}
    statement_lines: dict[str, list[int]] = {}
    for relative in relative_names:
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = list(range(1, 21))
        if relative == "static/course/labs/lab.py":
            lines = list(range(1, 19)) + [80, 184]
        source_lines = [""] * max(lines)
        for line in lines:
            source_lines[line - 1] = f"value_{line} = {line}"
        raw = (chr(10).join(source_lines) + chr(10)).encode()
        path.write_bytes(raw)
        inventory[relative] = {"size_bytes": len(raw), "sha256": _sha(raw)}
        statement_lines[relative] = lines
    inventory_digest = _sha(
        json.dumps(
            inventory, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode()
    )
    (repo / "committed-inventory.json").write_bytes(_raw(inventory))

    wheel = tmp_path / "llm_foundations_companion-0.1-py3-none-any.whl"
    wheel.write_bytes(b"deterministic wheel")
    receipt = tmp_path / "coverage"
    receipt.mkdir()
    installed_package = (
        receipt / "instrumented/site/llm_foundations_companion"
    )
    original_installed_package = (
        receipt / "locked/site/llm_foundations_companion"
    )
    shutil.copytree(package, installed_package)
    shutil.copytree(package, original_installed_package)
    refs: dict[str, dict[str, object]] = {}

    def put(path: str, value: Any, *, json_value: bool = True) -> dict[str, object]:
        target = receipt / path
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = _raw(value) if json_value else value
        target.write_bytes(raw)
        reference = {"path": path, "size_bytes": len(raw), "sha256": _sha(raw)}
        refs[path] = reference
        return reference

    def existing(path: str) -> dict[str, object]:
        raw = (receipt / path).read_bytes()
        reference = {"path": path, "size_bytes": len(raw), "sha256": _sha(raw)}
        refs[path] = reference
        return reference

    candidate = {
        "format": "s2-coverage-candidate-identity-v1",
        "repo": str(repo.resolve()),
        "commit": COMMIT,
        "tree": TREE,
        "clean": True,
        "status_porcelain_v1": "",
        "package_path": str(package.resolve()),
        "package_python_file_count": 54,
        "package_inventory_algorithm": "sha256-canonical-json-v1",
        "package_inventory_sha256": inventory_digest,
    }
    candidate_ref = put("candidate-identity.json", candidate)
    binding = {
        "format": "s2-coverage-path-binding-v2",
        "source_package": str(package.resolve()),
        "installed_package": str(installed_package.resolve()),
        "original_installed_package": str(original_installed_package.resolve()),
        "expected_python_file_count": 54,
        "source_python_file_count": 54,
        "installed_python_file_count": 54,
        "original_installed_python_file_count": 54,
        "source_only": [],
        "installed_only": [],
        "mismatched": [],
        "original_source_only": [],
        "original_installed_only": [],
        "original_mismatched": [],
        "files": inventory,
        "static_lab_python_files": LABS,
        "status": "PASS",
    }
    binding_ref = put("package-path-binding.json", binding)

    def executed(relative: str) -> list[int]:
        missing = 3 if under_threshold and relative == relative_names[0] else 1
        return statement_lines[relative][:-missing]

    def coverage_row(
        name: str,
        phase: str,
        values: dict[str, list[int]],
    ) -> dict[str, object]:
        target = receipt / f"raw/{name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        data = CoverageData(basename=str(target))
        data.set_context(phase)
        data.add_lines(
            {
                str((package / relative).resolve()): set(lines)
                for relative, lines in values.items()
            }
        )
        data.write()
        reference = existing(f"raw/{name}")
        return {
            "name": name,
            "size_bytes": reference["size_bytes"],
            "sha256": reference["sha256"],
            "phase": phase,
        }

    raw_rows = [
        coverage_row(
            ".coverage.fixture.pid900.unit",
            "unit",
            {relative: executed(relative) for relative in relative_names},
        ),
        coverage_row(
            ".coverage.fixture.pid1000.native_service",
            "native",
            {"__main__.py": [1], "service.py": [1]},
        ),
    ]
    raw_rows.extend(
        coverage_row(
            f".coverage.fixture.pid{1001 + index}.native_worker",
            "native",
            {"worker_main.py": [index + 1]},
        )
        for index in range(12)
    )
    raw_rows.extend(
        coverage_row(
            f".coverage.fixture.pid{1013 + index}.native_support",
            "native",
            {"module_00.py": [index + 1]},
        )
        for index in range(2)
    )
    raw_rows.append(
        coverage_row(
            ".coverage.fixture.pid2000.legacy_parent",
            "legacy",
            {
                relative: [statement_lines[relative][0]]
                for relative in (
                    "static/course/labs/data.py",
                    "static/course/labs/model.py",
                    "static/course/labs/test_lab.py",
                    "static/course/labs/tokenizer.py",
                )
            },
        )
    )
    raw_rows.extend(
        coverage_row(
            f".coverage.fixture.pid{2001 + index}.legacy_child",
            "legacy",
            {
                "static/course/labs/lab.py": (
                    sorted(
                        {index + 1, 184}
                        | ({80} if index < 5 else set())
                    )
                )
            },
        )
        for index in range(7)
    )
    if copied_native_raw:
        source_name = ".coverage.fixture.pid1001.native_worker"
        target_name = ".coverage.fixture.pid1002.native_worker"
        shutil.copyfile(
            receipt / f"raw/{source_name}",
            receipt / f"raw/{target_name}",
        )
        reference = existing(f"raw/{target_name}")
        target_row = next(row for row in raw_rows if row["name"] == target_name)
        target_row["size_bytes"] = reference["size_bytes"]
        target_row["sha256"] = reference["sha256"]
    combined_target = receipt / "raw/.coverage"
    combined = CoverageData(basename=str(combined_target))
    for row in raw_rows:
        raw_data = CoverageData(basename=str(receipt / f"raw/{row['name']}"))
        raw_data.read()
        combined.update(raw_data)
    if unrelated_combined:
        relative = relative_names[0]
        combined.add_lines(
            {
                str((package / relative).resolve()): set(
                    statement_lines[relative]
                )
            }
        )
    combined.write()
    combined_ref = existing("raw/.coverage")
    combined_data = CoverageData(basename=str(combined_target))
    combined_data.read()
    combined_executed = {
        relative: sorted(
            combined_data.lines(str((package / relative).resolve())) or []
        )
        for relative in relative_names
    }
    if invalid_raw:
        invalid_path = receipt / "raw/.coverage.fixture.pid900.unit"
        invalid_path.write_bytes(b"sealed but not CoverageData")
        reference = existing("raw/.coverage.fixture.pid900.unit")
        raw_rows[0]["size_bytes"] = reference["size_bytes"]
        raw_rows[0]["sha256"] = reference["sha256"]

    def audit_row(
        raw_row: dict[str, object],
        *,
        role: str,
        root_name: str,
        pid: int,
    ) -> dict[str, object]:
        data = CoverageData(basename=str(receipt / f"raw/{raw_row['name']}"))
        data.read()
        measured = sorted(data.measured_files())
        counts = {
            path: len(data.lines(path) or [])
            for path in measured
            if data.lines(path)
        }
        bindings = []
        for path in sorted(counts):
            relative = Path(path).relative_to(package).as_posix()
            lines = sorted(data.lines(path) or [])
            bindings.append(
                {
                    "path": path,
                    "root": root_name,
                    "package_relative_path": relative,
                    "executed_lines": len(lines),
                    "executed_line_numbers": lines,
                    "sha256": inventory[relative]["sha256"],
                }
            )
        return {
            "name": raw_row["name"],
            "size_bytes": raw_row["size_bytes"],
            "sha256": raw_row["sha256"],
            "pid": pid,
            "measured_files": measured,
            "contexts": sorted(data.measured_contexts()),
            "executed_line_counts": counts,
            "binding_failures": [],
            "role": role,
            "executed_path_bindings": bindings,
        }

    blocked_ids = sorted(gate_module.CAPABILITY_BLOCKED_IDS)
    native_authority_document = {
        "format": "llm-foundations-s2-native-development-v1",
        "denominator": 25,
        "cases": [
            {
                "case_id": f"S2-NATIVE-{number:03d}",
                "status": (
                    "BLOCKED"
                    if f"S2-NATIVE-{number:03d}" in blocked_ids
                    else "PASS"
                ),
            }
            for number in range(1, 26)
        ],
        "counts": {"PASS": 18, "FAIL": 0, "BLOCKED": 7, "NOT_RUN": 0},
        "supporting_tests": [
            {"test": f"support-{index}", "status": "PASS"}
            for index in range(3)
        ],
        "unknown_or_duplicate_tests": [],
        "discovery_error": None,
        "development_exclusions": {
            "enabled": True,
            "case_ids": blocked_ids,
            "reason": "test fixture",
        },
        "s2_native_development_gate": "FAIL",
    }
    native_authority_ref = put(
        "ordinary-authority/native-result.json",
        native_authority_document,
    )
    legacy_authority_document = {
        "format": "llm-foundations-s2-legacy-checks-v1",
        "status": "PASS",
        "tests": [
            {"test": f"legacy-{index}", "status": "PASS"}
            for index in range(7)
        ],
        "python": "3.12.3",
        "torch": "2.11.0+cpu",
        "product_qualification": "NOT_RUN",
    }
    legacy_authority_ref = put(
        "ordinary-authority/legacy-result.json",
        legacy_authority_document,
    )
    native_raw = [row for row in raw_rows if row["phase"] == "native"]
    service_names = [".coverage.fixture.pid1000.native_service"]
    worker_names = [
        f".coverage.fixture.pid{1001 + index}.native_worker"
        for index in range(12)
    ]
    support_names = [
        f".coverage.fixture.pid{1013 + index}.native_support"
        for index in range(2)
    ]
    native_audit = {
        "format": "s2-native-raw-process-audit-v3",
        "status": "PASS",
        "new_raw_file_count": 15,
        "service_raw_files": service_names,
        "worker_raw_files": worker_names,
        "parent_or_support_raw_files": support_names,
        "binding_failures": [],
        "contexts_ok": True,
        "roots": {
            "source": str(package.resolve()),
            "installed": str(installed_package.resolve()),
            "original_installed": str(original_installed_package.resolve()),
            "protected_labs": str((package / "static/course/labs").resolve()),
        },
        "files": [
            audit_row(
                row,
                role=(
                    "service"
                    if row["name"] in service_names
                    else "worker"
                    if row["name"] in worker_names
                    else "parent_or_support"
                ),
                root_name="source",
                pid=1000 + index,
            )
            for index, row in enumerate(native_raw)
        ],
    }
    if native_ghost:
        native_audit["service_raw_files"] = [".coverage.ghost"]
    if wrong_executed_root:
        native_audit["files"][0]["executed_path_bindings"][0]["root"] = (
            "installed"
        )
    if pid_mismatch:
        native_audit["files"][0]["pid"] += 10_000
    native_audit_ref = put("native-raw-process-audit.json", native_audit)
    legacy_raw = [row for row in raw_rows if row["phase"] == "legacy"]
    parent_names = [".coverage.fixture.pid2000.legacy_parent"]
    child_names = [
        f".coverage.fixture.pid{2001 + index}.legacy_child"
        for index in range(7)
    ]
    legacy_audit = {
        "format": "s2-legacy-raw-path-audit-v2",
        "status": "PASS",
        "new_raw_file_count": 8,
        "parent_raw_files": parent_names,
        "child_raw_files": child_names,
        "support_raw_files": [],
        "parent_raw_file_count": 1,
        "child_raw_file_count": 7,
        "support_raw_file_count": 0,
        "contexts_ok": True,
        "only_protected_execution": True,
        "required_executed_files": [
            f"static/course/labs/{name}.py"
            for name in ("data", "lab", "model", "test_lab", "tokenizer")
        ],
        "observed_executed_files": [
            f"static/course/labs/{name}.py"
            for name in ("data", "lab", "model", "test_lab", "tokenizer")
        ],
        "entrypoint_execution": {
            "main_first_body_line": 184,
            "train_first_body_line": 80,
            "child_raw_files_executing_main": child_names,
            "child_raw_files_executing_train": child_names[:5],
        },
        "binding_failures": [],
        "roots": native_audit["roots"],
        "files": [
            audit_row(
                row,
                role=(
                    "unittest_parent"
                    if row["name"] in parent_names
                    else "lab_child"
                ),
                root_name="protected_labs",
                pid=2000 + index,
            )
            for index, row in enumerate(legacy_raw)
        ],
    }
    if legacy_missing_child_entrypoint:
        legacy_audit["entrypoint_execution"]["child_raw_files_executing_main"] = (
            child_names[:-1]
        )
    legacy_audit_ref = put("legacy-raw-path-audit.json", legacy_audit)
    native_instrumented_document = {
        "format": "s2-instrumented-native-unittest-v1",
        "qualification": "coverage_only_modified_runtime",
        "started_at": "2026-10-01T00:00:00Z",
        "ended_at": "2026-10-01T00:00:01Z",
        "tests_run": 3,
        "successful": True,
        "known_app009_cases": [
            {"case_id": case_id, "status": "BLOCKED", "executed": False}
            for case_id in blocked_ids
        ],
        "known_app009_blocked_count": 7,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "expected_failures": 0,
        "unexpected_successes": 0,
        "tests": [
            {
                "test": f"instrumented.native.TestCase.test_{index}",
                "status": "PASS",
                "started_at": "2026-10-01T00:00:00Z",
                "ended_at": "2026-10-01T00:00:01Z",
            }
            for index in range(3)
        ],
    }
    native_instrumented_ref = put(
        "native-unittest-result.json",
        native_instrumented_document,
    )
    native_instrumented = {
        "format": "s2-instrumented-native-unittest-v1",
        "qualification": "coverage_only_modified_runtime",
        "successful": True,
        "tests_run": 3,
        "known_app009_blocked_count": 7,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "expected_failures": 0,
        "unexpected_successes": 0,
        "artifact": native_instrumented_ref,
    }
    legacy_stdout_ref = put(
        "legacy-instrumented.stdout.txt",
        b"seven protected tests passed",
        json_value=False,
    )
    legacy_stderr_ref = put(
        "legacy-instrumented.stderr.txt",
        b"",
        json_value=False,
    )
    legacy_test_ids = [
        item["test"]
        for item in legacy_authority_document["tests"]
    ]
    legacy_instrumented = {
        "format": "s2-instrumented-protected-legacy-unittest-v1",
        "qualification": "coverage_only_modified_legacy_runtime",
        "successful": True,
        "tests_run": 7,
        "test_ids": legacy_test_ids,
        "stdout_artifact": legacy_stdout_ref,
        "stderr_artifact": legacy_stderr_ref,
    }

    phase_result_refs: dict[str, dict[str, object]] = {}
    phase_member_refs: dict[str, list[dict[str, object]]] = {}
    cumulative_raw = 0
    for phase in ("unit", "native", "legacy"):
        phase_raw = [row for row in raw_rows if row["phase"] == phase]
        cumulative_raw += len(phase_raw)
        phase_result = {
            "format": "s2-coverage-measurement-v3",
            "phase": phase,
            "qualification": {
                "unit": "development_source_coverage_only",
                "native": "coverage_only_modified_runtime",
                "legacy": "coverage_only_modified_legacy_runtime",
            }[phase],
            "started_at": "2026-10-01T00:00:00Z",
            "ended_at": "2026-10-01T00:00:01Z",
            "exit_code": 0,
            "command_exit_code": 0,
            "source_stable": True,
            "raw_data_files_total": cumulative_raw,
            "new_raw_files": [
                {key: row[key] for key in ("name", "size_bytes", "sha256")}
                for row in phase_raw
            ],
            "ordinary_native_authority": None,
            "ordinary_legacy_authority": None,
            "native_process_audit_status": "PASS" if phase == "native" else None,
            "legacy_path_audit_status": "PASS" if phase == "legacy" else None,
            "instrumented_native": (
                native_instrumented if phase == "native" else None
            ),
            "instrumented_legacy": (
                legacy_instrumented if phase == "legacy" else None
            ),
        }
        phase_result_refs[phase] = put(f"{phase}-result.json", phase_result)
        members = [phase_result_refs[phase]] + [
            refs[f"raw/{row['name']}"] for row in phase_raw
        ]
        if phase == "unit" and extra_phase_raw:
            members.append(refs["raw/.coverage.fixture.pid1000.native_service"])
        if phase == "native":
            members += [
                native_audit_ref,
                native_authority_ref,
                native_instrumented_ref,
            ]
        if phase == "legacy":
            members += [
                legacy_audit_ref,
                legacy_authority_ref,
                legacy_stdout_ref,
                legacy_stderr_ref,
            ]
        phase_member_refs[phase] = members

    preparation = {
        "format": "s2-combined-coverage-preparation-v2",
        "status": "PASS",
        "measurements_run": False,
        "runtime_original_stable": True,
        "legacy_runtime_original_stable": True,
        "canonical_labs_stable": True,
        "package_path_binding": "PASS",
        "package_python_file_count": 54,
        "protected_lab_python_file_count": 6,
        "coverage_manifest_count": 106,
        "legacy_coverage_manifest_count": 106,
        "candidate": candidate,
        "wheel_sha256": _sha(wheel.read_bytes()),
    }
    preparation_ref = put("preparation-result.json", preparation)

    def make_index(name: str, members: list[dict[str, object]]) -> dict[str, object]:
        index_name = f"{name}-artifact-index.json"
        index_value = {
            "format": f"s2-coverage-{name}-artifact-index-v1",
            "created_at": "2026-10-01T00:00:02Z",
            "member_count": len(members),
            "members": sorted(members, key=lambda item: item["path"]),
        }
        index_ref = put(index_name, index_value)
        index_raw = (receipt / index_name).read_bytes()
        receipt_ref = put(
            f"{index_name}.sha256.json",
            {
                "format": "s2-artifact-index-self-hash-v1",
                "index": index_name,
                "size_bytes": len(index_raw),
                "sha256": _sha(index_raw),
            },
        )
        return {"index": index_ref, "receipt": receipt_ref}

    indexes = {
        "preparation": make_index(
            "preparation", [candidate_ref, binding_ref, preparation_ref]
        ),
        **{
            phase: make_index(phase, phase_member_refs[phase])
            for phase in ("unit", "native", "legacy")
        },
    }

    file_rows = []
    coverage_files: dict[str, object] = {}
    for relative in relative_names:
        path = str((package / relative).resolve())
        executed_lines = combined_executed[relative]
        missing_lines = sorted(set(statement_lines[relative]) - set(executed_lines))
        covered = len(executed_lines)
        excluded_lines = (
            [max(statement_lines[relative]) + 1]
            if exclusions and relative == relative_names[0]
            else []
        )
        excluded = len(excluded_lines)
        row = {
            "path": path,
            "package_relative_path": relative,
            **inventory[relative],
            "statements": 20,
            "covered": covered,
            "missing": 20 - covered,
            "excluded": excluded,
            "percent": covered * 100.0 / 20,
            "missing_lines": missing_lines,
        }
        file_rows.append(row)
        coverage_files[path] = {
            "executed_lines": executed_lines,
            "missing_lines": missing_lines,
            "excluded_lines": excluded_lines,
            "summary": {
                "covered_lines": covered,
                "num_statements": 20,
                "percent_covered": covered * 100.0 / 20,
                "missing_lines": 20 - covered,
                "excluded_lines": excluded,
            },
        }
    if fabricated_report_lines:
        first = str((package / relative_names[0]).resolve())
        coverage_files[first]["executed_lines"][-1] = 999
    if missing_files:
        file_rows = file_rows[:-missing_files]
        coverage_files = dict(list(coverage_files.items())[:-missing_files])

    totals = {
        "covered_lines": sum(row["covered"] for row in file_rows),
        "num_statements": sum(row["statements"] for row in file_rows),
        "missing_lines": sum(row["missing"] for row in file_rows),
        "excluded_lines": sum(row["excluded"] for row in file_rows),
    }
    totals["percent_covered"] = (
        95.0
        if under_threshold
        else totals["covered_lines"] * 100.0 / totals["num_statements"]
    )
    coverage_ref = put(
        "coverage.json",
        {"meta": {"format": 3}, "files": coverage_files, "totals": totals},
    )
    roster_ref = put(
        "raw-input-roster.json",
        {
            "format": "s2-coverage-raw-input-roster-v1",
            "status": "PASS",
            "raw_data_file_count": len(raw_rows),
            "files": raw_rows,
            "unexpected_files": [],
            "missing_files": [],
        },
    )
    phases: dict[str, object] = {}
    for phase in ("unit", "native", "legacy"):
        phase_raw = [row for row in raw_rows if row["phase"] == phase]
        phase_row: dict[str, object] = {
            "status": "PASS",
            "qualification": {
                "unit": "development_source_coverage_only",
                "native": "coverage_only_modified_runtime",
                "legacy": "coverage_only_modified_legacy_runtime",
            }[phase],
            "exit_code": 0,
            "command_exit_code": 0,
            "source_stable": True,
            "raw_data_file_count": len(phase_raw),
            "result_artifact": phase_result_refs[phase],
            "artifact_index": indexes[phase],
            "instrumented": (
                native_instrumented
                if phase == "native"
                else legacy_instrumented
                if phase == "legacy"
                else None
            ),
        }
        if phase == "native":
            phase_row.update(
                ordinary_authority={
                    "source_path": "/ordinary/native/result.json",
                    "artifact_path": native_authority_ref["path"],
                    "size_bytes": native_authority_ref["size_bytes"],
                    "sha256": native_authority_ref["sha256"],
                    "wheel_sha256": _sha(wheel.read_bytes()),
                    "aggregate": "FAIL_WITH_7_KNOWN_APP009_BLOCKED",
                    "counts": {"PASS": 18, "FAIL": 0, "BLOCKED": 7, "NOT_RUN": 0},
                    "supporting_tests": len(
                        native_authority_document["supporting_tests"]
                    ),
                    "excluded_case_ids": sorted(gate_module.CAPABILITY_BLOCKED_IDS),
                },
                process_audit_status="PASS",
            )
        if phase == "legacy":
            phase_row.update(
                ordinary_authority={
                    "source_path": "/ordinary/legacy/result.json",
                    "artifact_path": legacy_authority_ref["path"],
                    "size_bytes": legacy_authority_ref["size_bytes"],
                    "sha256": legacy_authority_ref["sha256"],
                    "tests": 7,
                    "status": "PASS",
                    "python": "3.12.3",
                    "torch": "2.11.0+cpu",
                    "product_qualification": "NOT_RUN",
                },
                path_audit_status="PASS",
            )
        phases[phase] = phase_row

    result = {
        "format": "s2-combined-coverage-result-v3",
        "status": "PASS",
        "threshold_percent": 95,
        "combine_exit_code": 0,
        "report_exit_code": 0,
        "json_exit_code": 0,
        "totals": totals,
        "reported_python_file_count": len(file_rows),
        "expected_python_file_count": 54,
        "complete_inventory": {
            "status": "PASS",
            "expected_file_count": 54,
            "reported_file_count": len(file_rows),
            "package_inventory_algorithm": "sha256-canonical-json-v1",
            "package_inventory_sha256": inventory_digest,
        },
        "zero_exclusions": {
            "status": "PASS" if not exclusions else "FAIL",
            "excluded_line_count": totals["excluded_lines"],
            "files_with_exclusions": [] if not exclusions else [file_rows[0]["path"]],
        },
        "candidate": candidate,
        "candidate_identity_artifact": candidate_ref,
        "wheel": {
            "path": str(wheel.resolve()),
            "size_bytes": wheel.stat().st_size,
            "sha256": _sha(wheel.read_bytes()),
        },
        "package_binding_artifact": binding_ref,
        "coverage_report_artifact": coverage_ref,
        "artifact_indexes": indexes,
        "phases": phases,
        "native_process_audit": {
            "status": "PASS",
            "service_raw_file_count": 1,
            "worker_raw_file_count": 12,
            "parent_or_support_raw_file_count": 2,
            "artifact": native_audit_ref,
        },
        "legacy_path_audit": {
            "status": "PASS",
            "parent_raw_file_count": 1,
            "child_raw_file_count": 7,
            "support_raw_file_count": 0,
            "entrypoint_execution": legacy_audit["entrypoint_execution"],
            "artifact": legacy_audit_ref,
        },
        "original_runtime_unchanged": True,
        "original_legacy_runtime_unchanged": True,
        "canonical_labs_unchanged": True,
        "source_stable": True,
        "package_binding": "PASS",
        "raw_data_file_count": len(raw_rows),
        "raw_data_files": raw_rows,
        "combined_data": combined_ref,
        "raw_input_roster_artifact": roster_ref,
        "files": file_rows,
        "protected_lab_files": [
            row for row in file_rows if row["package_relative_path"] in LABS
        ],
        "scope": SCOPE,
    }
    result_ref = put("result.json", result)
    final_members = sorted(refs.values(), key=lambda item: item["path"])
    evidence = {
        "format": "s2-combined-coverage-evidence-index-v1",
        "created_at": "2026-10-01T00:00:03Z",
        "member_count": len(final_members),
        "members": final_members,
    }
    evidence_raw = _raw(evidence)
    (receipt / "evidence-index.json").write_bytes(evidence_raw)
    (receipt / "evidence-index.sha256.json").write_bytes(
        _raw(
            {
                "format": "s2-artifact-index-self-hash-v1",
                "index": "evidence-index.json",
                "size_bytes": len(evidence_raw),
                "sha256": _sha(evidence_raw),
            }
        )
    )
    assert result_ref["path"] == "result.json"
    return repo, wheel, receipt / "result.json"


def _verify(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    wheel: Path,
    result: Path,
) -> dict[str, object]:
    monkeypatch.setattr(
        gate_module,
        "git",
        lambda _repo, *args: (
            COMMIT
            if args == ("rev-parse", "HEAD")
            else TREE
            if args == ("rev-parse", "HEAD^{tree}")
            else ""
            if args[:2] == ("status", "--porcelain=v1")
            else "0"
        ),
    )
    monkeypatch.setattr(
        gate_module,
        "_committed_package_inventory",
        lambda candidate_repo, _commit: json.loads(
            (candidate_repo / "committed-inventory.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    return gate_module.verify_complete_source_coverage(
        result, repo=repo, commit=COMMIT, tree=TREE, wheel=wheel
    )


def test_complete_source_coverage_accepts_exact_sealed_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, wheel, result = _fixture(tmp_path)
    observed = _verify(monkeypatch, repo, wheel, result)
    assert observed["package_python_file_count"] == 54
    assert observed["coverage_percent"] >= 95.0


def test_missing_coverage_receipt_is_explicitly_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate_module, "git", lambda *_args: "0")
    reports = tmp_path / "reports"
    reports.mkdir()
    gate = gate_module.Gate(tmp_path, reports)
    wheel = tmp_path / "candidate.whl"
    wheel.write_bytes(b"wheel")
    row = gate_module.record_complete_source_coverage(
        gate,
        None,
        repo=tmp_path,
        commit=COMMIT,
        tree=TREE,
        wheel=wheel,
    )
    assert row["status"] == "BLOCKED"
    assert row["gate_id"] == "complete-source-coverage"
    failed = gate_module.record_complete_source_coverage(
        gate,
        tmp_path / "absent-result.json",
        repo=tmp_path,
        commit=COMMIT,
        tree=TREE,
        wheel=wheel,
    )
    assert failed["status"] == "FAIL"
    assert failed["gate_id"] == "complete-source-coverage"


@pytest.mark.parametrize(
    ("variant", "match"),
    [
        ("rounded", "below the exact 95 percent"),
        ("missing", "Coverage file count differs"),
        ("exclusions", "counts or exclusions differ"),
    ],
)
def test_semantic_coverage_forgeries_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    match: str,
) -> None:
    repo, wheel, result = _fixture(
        tmp_path,
        under_threshold=variant == "rounded",
        missing_files=6 if variant == "missing" else 0,
        exclusions=variant == "exclusions",
    )
    with pytest.raises(ValueError, match=match):
        _verify(monkeypatch, repo, wheel, result)


def test_stale_source_and_wheel_identities_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, wheel, result = _fixture(tmp_path / "source")
    next((repo / "companion/src/llm_foundations_companion").rglob("*.py")).write_text(
        "changed = True" + chr(10), encoding="utf-8"
    )
    with pytest.raises(
        ValueError,
        match="captured Git commit|candidate identity is stale|file identity differs",
    ):
        _verify(monkeypatch, repo, wheel, result)

    repo, wheel, result = _fixture(tmp_path / "wheel")
    wheel.write_bytes(b"different wheel")
    with pytest.raises(ValueError, match="wheel identity differs"):
        _verify(monkeypatch, repo, wheel, result)


@pytest.mark.parametrize(
    ("keyword", "match"),
    [
        ("invalid_raw", "Raw coverage database is invalid"),
        ("native_ghost", "role partitions differ"),
        ("legacy_missing_child_entrypoint", "child entrypoint execution differs"),
        ("extra_phase_raw", "artifact index raw roster differs"),
        ("fabricated_report_lines", "counts or exclusions differ"),
        ("wrong_executed_root", "outside its declared root"),
        ("unrelated_combined", "exact raw-data union"),
        ("pid_mismatch", "PID differs from its filename"),
        ("copied_native_raw", "raw identities are not distinct"),
    ],
)
def test_raw_database_and_process_audit_forgeries_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    keyword: str,
    match: str,
) -> None:
    repo, wheel, result = _fixture(tmp_path, **{keyword: True})
    with pytest.raises(ValueError, match=match):
        _verify(monkeypatch, repo, wheel, result)


def test_uncommitted_or_ignored_package_file_cannot_enter_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, wheel, result = _fixture(tmp_path)
    (
        repo
        / "companion/src/llm_foundations_companion/ignored_addition.py"
    ).write_text("ignored = True" + chr(10), encoding="utf-8")
    with pytest.raises(ValueError, match="captured Git commit"):
        _verify(monkeypatch, repo, wheel, result)


def test_raw_artifact_drift_breaks_final_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, wheel, result = _fixture(tmp_path)
    (
        result.parent / "raw/.coverage.fixture.pid1001.native_worker"
    ).write_bytes(b"drift")
    with pytest.raises(ValueError, match="Artifact identity differs"):
        _verify(monkeypatch, repo, wheel, result)
