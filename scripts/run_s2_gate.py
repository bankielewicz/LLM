"""Run the bounded S2 source, installed-service and native-model gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import textwrap
import time
import uuid

from check_custody import audit as custody_audit, canonical, sha256
from check_s2_authority import audit as s2_authority_audit
from run_s0_gate import authoring_environment, git, qualified_python, source_manifest, write_new
from run_s1_gate import unit_denominator
from run_s2_native_checks import CASE_RE as NATIVE_CASE_RE
from run_s2_native_checks import EXPECTED_SUPPORT_TESTS, TEST_CASE_IDS


PLAN = (
    "python-runtime",
    "authoring-environment",
    "s2-authority-before",
    "custody-before",
    "runtime-lock-identities",
    "s0-regression-gate",
    "bundled-contracts",
    "bundled-reader",
    "s2-wsl-unit-tests",
    "native-windows-control",
    "unit-denominator",
    "runtime-preinstall-state",
    "build-wheel",
    "wheel-source-binding",
    "complete-source-coverage",
    "install-wheel",
    "runtime-environment",
    "installed-service-regressions",
    "legacy-lab-checks",
    "native-model-checks",
    "native-denominator",
    "s2-authority-after",
    "custody-after",
    "candidate-stability",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")
_RAW_PID_RE = re.compile(r"(?:^|[.])pid([1-9][0-9]*)(?:[.]|$)")
_COVERAGE_FORMAT = "s2-combined-coverage-result-v4"
_EXPECTED_NATIVE_ROLE_COUNTS = {
    "runner_parent": 1,
    "service": 1,
    "preflight_support": 1,
    "cli_support": 1,
    "worker": 19,
}
_EXPECTED_NATIVE_OPERATION_COUNTS = {
    "tokenizer_train": 4,
    "tiny_train": 5,
    "tiny_resume": 2,
    "evaluate": 1,
    "context_preview": 4,
    "generate": 3,
}
_COVERAGE_INDEX_FORMAT = "s2-combined-coverage-evidence-index-v1"
_INDEX_RECEIPT_FORMAT = "s2-artifact-index-self-hash-v1"
_PACKAGE_PATH = PurePosixPath("companion/src/llm_foundations_companion")
_PROTECTED_LABS = (
    "static/course/labs/data.py",
    "static/course/labs/lab.py",
    "static/course/labs/make_demo_data.py",
    "static/course/labs/model.py",
    "static/course/labs/test_lab.py",
    "static/course/labs/tokenizer.py",
)
_COVERAGE_FIELDS = frozenset(
    {
        "format",
        "status",
        "threshold_percent",
        "combine_exit_code",
        "report_exit_code",
        "json_exit_code",
        "totals",
        "reported_python_file_count",
        "expected_python_file_count",
        "complete_inventory",
        "zero_exclusions",
        "candidate",
        "candidate_identity_artifact",
        "wheel",
        "package_binding_artifact",
        "coverage_report_artifact",
        "artifact_indexes",
        "phases",
        "native_process_audit",
        "legacy_path_audit",
        "original_runtime_unchanged",
        "original_legacy_runtime_unchanged",
        "canonical_labs_unchanged",
        "source_stable",
        "package_binding",
        "raw_data_file_count",
        "raw_data_files",
        "combined_data",
        "raw_input_roster_artifact",
        "files",
        "protected_lab_files",
        "scope",
    }
)


class Gate:
    def __init__(self, repo: Path, reports: Path) -> None:
        self.repo = repo
        self.reports = reports
        self.rows: list[dict[str, object]] = []
        self.env = os.environ.copy()
        self.env.update(
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONUTF8="1",
            PYTHONPATH=str(repo / "companion/src"),
        )
        self.env["SOURCE_DATE_EPOCH"] = git(repo, "show", "-s", "--format=%ct", "HEAD")

    def observe(self, name: str, action: object) -> dict[str, object]:
        started = time.time()
        try:
            value = action()
            output = {
                "gate_id": name,
                "status": "PASS",
                "seconds": round(time.time() - started, 3),
                "observation": value,
            }
        except BaseException as exc:
            output = {
                "gate_id": name,
                "status": "FAIL",
                "seconds": round(time.time() - started, 3),
                "reason": type(exc).__name__ + ": " + str(exc),
            }
        self.rows.append(output)
        print(f"{name}: {output['status']}", flush=True)
        return output

    def command(
        self,
        name: str,
        argv: list[str],
        *,
        timeout: int = 600,
        announce: bool = True,
    ) -> dict[str, object]:
        started = time.time()
        try:
            result = subprocess.run(
                argv,
                cwd=self.repo,
                env=self.env,
                capture_output=True,
                timeout=timeout,
            )
            stdout, stderr, code = result.stdout, result.stderr, result.returncode
            status = "PASS" if code == 0 else "FAIL"
            reason = None if code == 0 else f"Process exited {code}"
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = exc.stdout or b"", exc.stderr or b""
            code, status, reason = None, "BLOCKED", str(exc)
        except OSError as exc:
            stdout, stderr = b"", str(exc).encode()
            code, status, reason = None, "BLOCKED", str(exc)
        output: dict[str, object] = {
            "gate_id": name,
            "status": status,
            "command": argv,
            "exit_code": code,
            "seconds": round(time.time() - started, 3),
        }
        if reason:
            output["reason"] = reason
        for label, raw in (("stdout", stdout), ("stderr", stderr)):
            path = self.reports / f"{name}.{label}.txt"
            path.write_bytes(raw)
            output[label] = {
                "path": path.name,
                "size_bytes": len(raw),
                "sha256": sha256(raw),
            }
        self.rows.append(output)
        if announce:
            print(f"{name}: {status}", flush=True)
        return output

    def blocked(self, name: str, reason: str) -> dict[str, object]:
        output = {"gate_id": name, "status": "BLOCKED", "reason": reason}
        self.rows.append(output)
        print(f"{name}: BLOCKED", flush=True)
        return output


RUNTIME_PREINSTALL_PROBE = textwrap.dedent(
    r"""
    import base64
    import csv
    import hashlib
    import importlib.metadata as metadata
    import importlib.util
    import io
    import json
    from pathlib import Path, PurePosixPath
    import re
    import sys
    import sysconfig

    def normalize(value):
        return re.sub(r"[-_.]+", "-", value).lower()

    allow_existing = sys.argv[1] == "1"
    purelib = Path(sysconfig.get_path("purelib")).resolve(strict=True)
    package = purelib / "llm_foundations_companion"
    matching_metadata = [
        path
        for path in purelib.iterdir()
        if path.is_dir()
        and (
            path.name.endswith(".dist-info")
            or path.name.endswith(".egg-info")
        )
        and normalize(path.name.rsplit(".", 1)[0]).startswith(
            "llm-foundations-companion-"
        )
    ]
    spec_present = importlib.util.find_spec("llm_foundations_companion") is not None
    try:
        distribution = metadata.distribution("llm-foundations-companion")
    except metadata.PackageNotFoundError:
        distribution = None

    if distribution is None:
        if spec_present or package.exists() or matching_metadata:
            raise RuntimeError(
                "The runtime has an incomplete or unowned companion installation"
            )
        print(json.dumps({
            "companion_present": False,
            "python": sys.version,
            "executable": sys.executable,
            "purelib": str(purelib),
        }, sort_keys=True))
        raise SystemExit(0)

    if not allow_existing:
        raise RuntimeError(
            "The runtime already contains the companion; pass "
            "--allow-existing-companion only to audit and replace a prior candidate"
        )
    if not spec_present or not package.is_dir() or package.is_symlink():
        raise RuntimeError("Installed companion package root is missing or unsafe")

    dist_root = Path(distribution._path).resolve(strict=True)
    if dist_root.parent != purelib or dist_root.is_symlink():
        raise RuntimeError("Installed distribution metadata root is unsafe")
    if matching_metadata != [dist_root]:
        raise RuntimeError("Installed companion metadata roots are ambiguous")
    if dist_root.suffix != ".dist-info":
        raise RuntimeError("Installed companion is not a wheel distribution")

    record_path = dist_root / "RECORD"
    if not record_path.is_file() or record_path.is_symlink():
        raise RuntimeError("Installed companion RECORD is missing or unsafe")
    rows = list(csv.reader(io.StringIO(
        record_path.read_text(encoding="utf-8"), newline=""
    )))
    if not rows or any(len(row) != 3 or not row[0] for row in rows):
        raise RuntimeError("Installed companion RECORD has an invalid row")
    if len(rows) != len({row[0] for row in rows}):
        raise RuntimeError("Installed companion RECORD has duplicate paths")

    observed = set()
    for root in (package, dist_root):
        for target in root.rglob("*"):
            if target.is_symlink():
                raise RuntimeError("Installed distribution contains a symlink")
            if target.is_dir():
                continue
            if not target.is_file():
                raise RuntimeError("Installed distribution contains a non-file entry")
            observed.add(target.relative_to(purelib).as_posix())
    recorded = {row[0] for row in rows}
    if recorded != observed:
        raise RuntimeError(
            "Installed RECORD does not describe the exact companion distribution"
        )

    files = []
    for name, digest, size in rows:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Installed RECORD path escapes purelib")
        target = purelib.joinpath(*relative.parts)
        raw = target.read_bytes()
        actual_digest = hashlib.sha256(raw).hexdigest()
        if digest:
            algorithm, encoded = digest.split("=", 1)
            expected = base64.urlsafe_b64encode(
                hashlib.sha256(raw).digest()
            ).rstrip(b"=").decode("ascii")
            if algorithm != "sha256" or encoded != expected:
                raise RuntimeError("Installed RECORD digest differs: " + name)
        if size and int(size) != len(raw):
            raise RuntimeError("Installed RECORD size differs: " + name)
        files.append({
            "path": name,
            "size_bytes": len(raw),
            "sha256": actual_digest,
            "record_digest_present": bool(digest),
            "record_size_present": bool(size),
        })

    import llm_foundations_companion
    version = distribution.version
    if (
        distribution.metadata["Name"] != "llm-foundations-companion"
        or version != llm_foundations_companion.__version__
    ):
        raise RuntimeError("Installed package and distribution versions differ")
    print(json.dumps({
        "companion_present": True,
        "prior_distribution": {
            "name": distribution.metadata["Name"],
            "version": version,
            "package_root": str(package),
            "metadata_root": str(dist_root),
            "record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
            "files": files,
        },
        "python": sys.version,
        "executable": sys.executable,
        "purelib": str(purelib),
    }, sort_keys=True))
    """
)


def runtime_preinstall(
    python: Path, *, allow_existing_companion: bool
) -> dict[str, object]:
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            RUNTIME_PREINSTALL_PROBE,
            "1" if allow_existing_companion else "0",
        ],
        capture_output=True,
        timeout=30,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            "Runtime preinstall audit failed: "
            + (result.stderr or result.stdout or f"exit {result.returncode}")[:4_000]
        )
    return json.loads(result.stdout)


def launcher(path: Path, parser: argparse.ArgumentParser, label: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    absolute = expanded.absolute()
    if not absolute.is_file():
        parser.error(f"{label} is not a file: {absolute}")
    return absolute


def native_denominator(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != "llm-foundations-s2-native-development-v1":
        raise ValueError("Native result format differs")
    cases = value.get("cases")
    expected = [f"S2-NATIVE-{number:03d}" for number in range(1, 26)]
    if (
        value.get("denominator") != 25
        or not isinstance(cases, list)
        or len(cases) != 25
        or [item.get("case_id") for item in cases] != expected
    ):
        raise ValueError("Native S2 denominator identities or order differ")
    canonical_acceptance = value.get("canonical_acceptance", {})
    mapped = value.get("mapped_s2_acceptance", {})
    if (
        canonical_acceptance.get("execution_units") != 605
        or canonical_acceptance.get("passed") != 0
        or canonical_acceptance.get("not_run") != 605
        or canonical_acceptance.get("product_qualification") != "NOT_RUN"
    ):
        raise ValueError("Full acceptance denominator was promoted or changed")
    if (
        mapped.get("execution_units") != 20
        or mapped.get("passed") != 0
        or mapped.get("not_run") != 20
    ):
        raise ValueError("Mapped S2 acceptance units were promoted or changed")
    supporting = value.get("supporting_tests")
    exclusions = value.get("development_exclusions")
    if (
        any(not isinstance(item, dict) or item.get("status") != "PASS" for item in cases)
        or value.get("counts")
        != {"PASS": 25, "FAIL": 0, "BLOCKED": 0, "NOT_RUN": 0}
        or not isinstance(supporting, list)
        or [item.get("test") for item in supporting] != list(EXPECTED_SUPPORT_TESTS)
        or any(
            not isinstance(item, dict) or item.get("status") != "PASS"
            for item in supporting
        )
        or value.get("unknown_or_duplicate_tests") != []
        or value.get("discovery_error") is not None
        or exclusions != {"enabled": False, "case_ids": [], "reason": None}
        or value.get("s2_native_development_gate") != "PASS"
    ):
        raise ValueError("Native S2 denominator is not exactly 25 PASS")
    return {
        "status": "PASS",
        "denominator": 25,
        "passed": 25,
        "supporting_tests": len(EXPECTED_SUPPORT_TESTS),
        "canonical_acceptance_not_run": 605,
        "mapped_s2_acceptance_not_run": 20,
        "profile": value.get("profile"),
    }

def _required(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _strict_json(path: Path) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"Duplicate JSON key in {path.name}: {key}")
            value[key] = item
        return value

    def nonfinite(value: str) -> object:
        raise ValueError(f"Non-finite JSON value in {path.name}: {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"Non-finite JSON value in {path.name}: {value}")
        return parsed

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=nonfinite,
        parse_float=finite_float,
    )


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _artifact_target(root: Path, reference: object) -> Path:
    _required(
        isinstance(reference, dict)
        and set(reference) == {"path", "size_bytes", "sha256"},
        "Artifact reference schema differs",
    )
    name = reference["path"]
    _required(isinstance(name, str) and name and chr(92) not in name, "Unsafe artifact path")
    relative = PurePosixPath(name)
    _required(
        not relative.is_absolute()
        and ".." not in relative.parts
        and "." not in relative.parts
        and ":" not in relative.parts[0],
        "Artifact path escapes the receipt root",
    )
    target = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        _required(not current.is_symlink(), f"Artifact path uses a symlink: {name}")
    _required(target.is_file(), f"Artifact is missing: {name}")
    raw = target.read_bytes()
    _required(
        _integer(reference["size_bytes"])
        and reference["size_bytes"] == len(raw)
        and isinstance(reference["sha256"], str)
        and _SHA256_RE.fullmatch(reference["sha256"]) is not None
        and reference["sha256"] == sha256(raw),
        f"Artifact identity differs: {name}",
    )
    return target


def _index_members(
    root: Path,
    value: object,
    expected_format: str,
) -> dict[str, dict[str, object]]:
    _required(
        isinstance(value, dict)
        and set(value) == {"format", "created_at", "member_count", "members"}
        and value["format"] == expected_format
        and isinstance(value["created_at"], str)
        and value["created_at"],
        "Artifact index schema differs",
    )
    members = value["members"]
    _required(
        isinstance(members, list)
        and _integer(value["member_count"])
        and value["member_count"] == len(members),
        "Artifact index member count differs",
    )
    output: dict[str, dict[str, object]] = {}
    for reference in members:
        target = _artifact_target(root, reference)
        name = reference["path"]
        _required(name not in output, f"Duplicate artifact index member: {name}")
        output[name] = reference
        _required(target.relative_to(root).as_posix() == name, "Artifact path normalized")
    _required(list(output) == sorted(output), "Artifact index members are not sorted")
    return output


def _member(
    root: Path,
    reference: object,
    members: dict[str, dict[str, object]],
) -> Path:
    target = _artifact_target(root, reference)
    _required(
        members.get(reference["path"]) == reference,
        f"Artifact is absent from the evidence index: {reference['path']}",
    )
    return target


def _load_child_index(
    root: Path,
    pair: object,
    name: str,
    final_members: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    _required(
        isinstance(pair, dict) and set(pair) == {"index", "receipt"},
        f"{name} artifact index references differ",
    )
    index_path = _member(root, pair["index"], final_members)
    receipt_path = _member(root, pair["receipt"], final_members)
    expected_name = f"{name}-artifact-index.json"
    _required(
        index_path.name == expected_name
        and receipt_path.name == f"{expected_name}.sha256.json",
        f"{name} artifact index filenames differ",
    )
    receipt = _strict_json(receipt_path)
    raw = index_path.read_bytes()
    _required(
        isinstance(receipt, dict)
        and set(receipt) == {"format", "index", "size_bytes", "sha256"}
        and receipt["format"] == _INDEX_RECEIPT_FORMAT
        and receipt["index"] == expected_name
        and receipt["size_bytes"] == len(raw)
        and receipt["sha256"] == sha256(raw),
        f"{name} artifact index self-hash differs",
    )
    members = _index_members(
        root,
        _strict_json(index_path),
        f"s2-coverage-{name}-artifact-index-v1",
    )
    for path, reference in members.items():
        _required(
            final_members.get(path) == reference,
            f"{name} index member is absent from the final index: {path}",
        )
    return members


def _directory_inventory(root: Path) -> dict[str, dict[str, object]]:
    _required(root.is_dir() and not root.is_symlink(), f"Package root is unsafe: {root}")
    inventory: dict[str, dict[str, object]] = {}
    paths = sorted(root.rglob("*.py"), key=lambda item: item.relative_to(root).as_posix())
    for path in paths:
        _required(path.is_file() and not path.is_symlink(), f"Package inventory is unsafe: {path}")
        raw = path.read_bytes()
        inventory[path.relative_to(root).as_posix()] = {
            "size_bytes": len(raw),
            "sha256": sha256(raw),
        }
    return inventory


def _committed_package_inventory(
    repo: Path,
    commit: str,
) -> dict[str, dict[str, object]]:
    result = subprocess.run(
        [
            "git", "-C", str(repo), "ls-tree", "-r", "-z", commit, "--",
            _PACKAGE_PATH.as_posix(),
        ],
        capture_output=True,
    )
    _required(result.returncode == 0, "Cannot read committed package inventory")
    inventory: dict[str, dict[str, object]] = {}
    prefix = _PACKAGE_PATH.as_posix() + "/"
    for entry in result.stdout.split(bytes([0])):
        if not entry:
            continue
        metadata, raw_path = entry.split(bytes([9]), 1)
        mode, kind, object_id = metadata.split(b" ")
        path = raw_path.decode("utf-8", "strict")
        if not path.endswith(".py"):
            continue
        _required(
            path.startswith(prefix)
            and kind == b"blob"
            and mode in {b"100644", b"100755"},
            "Committed package contains an unsafe Python entry",
        )
        relative = path[len(prefix):]
        _required(relative not in inventory, "Committed package inventory is duplicated")
        blob = subprocess.run(
            ["git", "-C", str(repo), "cat-file", "blob", object_id.decode("ascii")],
            capture_output=True,
        )
        _required(blob.returncode == 0, f"Cannot read committed package blob: {relative}")
        inventory[relative] = {
            "size_bytes": len(blob.stdout),
            "sha256": sha256(blob.stdout),
        }
    return dict(sorted(inventory.items()))


def _live_package(
    repo: Path,
    commit: str,
) -> tuple[Path, dict[str, dict[str, object]], str]:
    package = (repo / Path(*_PACKAGE_PATH.parts)).resolve(strict=True)
    inventory = _directory_inventory(package)
    _required(
        inventory == _committed_package_inventory(repo, commit),
        "Live package inventory differs from the captured Git commit",
    )
    _required(inventory, "Source package has no Python files")
    digest = sha256(
        json.dumps(
            inventory,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    return package, inventory, digest


def _raw_member(
    root: Path,
    row: object,
    final_members: dict[str, dict[str, object]],
    phase_members: dict[str, dict[str, object]],
) -> Path:
    _required(
        isinstance(row, dict)
        and set(row) == {"name", "size_bytes", "sha256", "phase"}
        and isinstance(row["name"], str)
        and row["name"].startswith(".coverage.")
        and "/" not in row["name"],
        "Raw coverage row differs",
    )
    matches = [
        reference
        for path, reference in final_members.items()
        if PurePosixPath(path).name == row["name"]
        and reference["size_bytes"] == row["size_bytes"]
        and reference["sha256"] == row["sha256"]
    ]
    _required(len(matches) == 1, f"Raw coverage artifact is ambiguous: {row['name']}")
    _required(
        phase_members.get(matches[0]["path"]) == matches[0],
        f"Raw coverage artifact is absent from its phase index: {row['name']}",
    )
    return _member(root, matches[0], final_members)


def _coverage_data(path: Path) -> dict[str, object]:
    try:
        from coverage import CoverageData
        from coverage.exceptions import CoverageException
    except ImportError as exc:
        raise ValueError("Coverage.py is required to verify raw coverage data") from exc
    try:
        data = CoverageData(basename=str(path))
        data.read()
        measured = sorted(data.measured_files())
        contexts = sorted(data.measured_contexts())
        lines = {
            name: sorted(data.lines(name) or [])
            for name in measured
        }
    except CoverageException as exc:
        raise ValueError(f"Raw coverage database is invalid: {path.name}") from exc
    for name, values in lines.items():
        _required(
            isinstance(name, str)
            and name
            and values == sorted(set(values))
            and all(_integer(line) and line > 0 for line in values),
            f"Raw coverage line data differs: {path.name}",
        )
    return {"measured_files": measured, "contexts": contexts, "lines": lines}


def _audit_raw_row(
    item: object,
    roster_row: dict[str, object],
    raw_data: dict[str, object],
    inventory: dict[str, dict[str, object]],
    *,
    context: str,
    roles: set[str],
    roots: set[str],
    root_paths: dict[str, Path],
) -> dict[str, set[int]]:
    expected_keys = {
        "name",
        "size_bytes",
        "sha256",
        "pid",
        "measured_files",
        "contexts",
        "executed_line_counts",
        "binding_failures",
        "role",
        "executed_path_bindings",
    }
    _required(
        isinstance(item, dict)
        and set(item) == expected_keys
        and item["name"] == roster_row["name"]
        and item["size_bytes"] == roster_row["size_bytes"]
        and item["sha256"] == roster_row["sha256"]
        and _integer(item["pid"])
        and item["pid"] > 0
        and item["measured_files"] == raw_data["measured_files"]
        and item["contexts"] == [context]
        and raw_data["contexts"] == [context]
        and item["binding_failures"] == []
        and item["role"] in roles
        and isinstance(item["executed_line_counts"], dict)
        and isinstance(item["executed_path_bindings"], list),
        f"{context} raw process audit row differs",
    )
    match = _RAW_PID_RE.search(roster_row["name"])
    _required(
        match is not None and int(match.group(1)) == item["pid"],
        f"{context} raw process PID differs from its filename",
    )
    actual_counts = {
        path: len(lines)
        for path, lines in raw_data["lines"].items()
        if lines
    }
    _required(
        item["executed_line_counts"] == actual_counts,
        f"{context} raw executed-line counts differ",
    )
    by_path: dict[str, dict[str, object]] = {}
    by_relative: dict[str, set[int]] = {}
    binding_keys = {
        "path",
        "root",
        "package_relative_path",
        "executed_lines",
        "executed_line_numbers",
        "sha256",
    }
    for binding in item["executed_path_bindings"]:
        _required(
            isinstance(binding, dict)
            and set(binding) == binding_keys
            and binding["path"] in actual_counts
            and binding["path"] not in by_path
            and binding["root"] in roots
            and binding["package_relative_path"] in inventory
            and binding["executed_lines"] == actual_counts[binding["path"]]
            and binding["executed_line_numbers"]
            == raw_data["lines"][binding["path"]]
            and binding["sha256"]
            == inventory[binding["package_relative_path"]]["sha256"],
            f"{context} raw executed-path binding differs",
        )
        declared_root = root_paths[binding["root"]]
        declared_target = Path(binding["path"])
        _required(
            declared_root.is_absolute()
            and declared_target.is_absolute()
            and declared_root.is_dir()
            and not declared_root.is_symlink()
            and declared_target.is_file()
            and not declared_target.is_symlink(),
            f"{context} executed path is missing or unsafe",
        )
        resolved_root = declared_root.resolve(strict=True)
        resolved_target = declared_target.resolve(strict=True)
        _required(
            resolved_target.is_relative_to(resolved_root),
            f"{context} executed path is outside its declared root",
        )
        expected_relative = binding["package_relative_path"]
        if binding["root"] == "protected_labs":
            expected_relative = PurePosixPath(expected_relative).relative_to(
                "static/course/labs"
            ).as_posix()
        _required(
            resolved_target.relative_to(resolved_root).as_posix()
            == expected_relative,
            f"{context} executed path mapping differs",
        )
        executed_raw = resolved_target.read_bytes()
        _required(
            len(executed_raw)
            == inventory[binding["package_relative_path"]]["size_bytes"]
            and sha256(executed_raw)
            == inventory[binding["package_relative_path"]]["sha256"],
            f"{context} executed path bytes differ from the live candidate",
        )
        by_path[binding["path"]] = binding
        by_relative.setdefault(binding["package_relative_path"], set()).update(
            binding["executed_line_numbers"]
        )
    _required(
        set(by_path) == set(actual_counts),
        f"{context} raw executed paths are not completely bound",
    )
    return by_relative


def _combined_coverage_analysis(
    path: Path,
    package: Path,
    inventory: dict[str, dict[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    try:
        from coverage import Coverage
        from coverage.exceptions import CoverageException
    except ImportError as exc:
        raise ValueError("Coverage.py is required to analyze combined coverage") from exc
    try:
        coverage = Coverage(data_file=str(path), config_file=False)
        coverage.set_option("report:exclude_lines", [])
        coverage.load()
        data = coverage.get_data()
        expected_paths = {
            str((package / relative).resolve())
            for relative in inventory
        }
        _required(
            set(data.measured_files()) == expected_paths,
            "Combined coverage database source roster differs",
        )
        analyses: dict[str, dict[str, object]] = {}
        for relative in inventory:
            source = str((package / relative).resolve())
            _, statements, excluded, missing, _ = coverage.analysis2(source)
            analyses[relative] = {
                "statements": sorted(statements),
                "excluded": sorted(excluded),
                "missing": sorted(missing),
                "executed": sorted(set(statements) - set(missing)),
            }
    except CoverageException as exc:
        raise ValueError("Combined coverage database cannot be analyzed") from exc
    return analyses, _coverage_data(path)


def verify_complete_source_coverage(
    result_path: Path,
    *,
    repo: Path,
    commit: str,
    tree: str,
    wheel: Path,
) -> dict[str, object]:
    _required(not result_path.is_symlink(), "Coverage result cannot be a symlink")
    result_path = result_path.resolve(strict=True)
    root = result_path.parent
    _required(result_path.name == "result.json", "Coverage result must be result.json")
    result = _strict_json(result_path)
    _required(
        isinstance(result, dict)
        and set(result) == _COVERAGE_FIELDS
        and result["format"] == _COVERAGE_FORMAT,
        "Coverage result schema differs",
    )

    index_path = root / "evidence-index.json"
    receipt_path = root / "evidence-index.sha256.json"
    _required(
        index_path.is_file()
        and receipt_path.is_file()
        and not index_path.is_symlink()
        and not receipt_path.is_symlink(),
        "Coverage final evidence index is missing or unsafe",
    )
    index_raw = index_path.read_bytes()
    receipt = _strict_json(receipt_path)
    _required(
        isinstance(receipt, dict)
        and set(receipt) == {"format", "index", "size_bytes", "sha256"}
        and receipt["format"] == _INDEX_RECEIPT_FORMAT
        and receipt["index"] == "evidence-index.json"
        and receipt["size_bytes"] == len(index_raw)
        and receipt["sha256"] == sha256(index_raw),
        "Coverage final evidence index self-hash differs",
    )
    final_members = _index_members(root, _strict_json(index_path), _COVERAGE_INDEX_FORMAT)
    _required(
        final_members.get("result.json")
        == {
            "path": "result.json",
            "size_bytes": len(result_path.read_bytes()),
            "sha256": sha256(result_path.read_bytes()),
        },
        "Coverage result is absent from the final evidence index",
    )
    indexes = result["artifact_indexes"]
    _required(
        isinstance(indexes, dict)
        and set(indexes) == {"preparation", "unit", "native", "legacy"},
        "Coverage phase indexes differ",
    )
    child = {
        name: _load_child_index(root, indexes[name], name, final_members)
        for name in ("preparation", "unit", "native", "legacy")
    }

    package, inventory, inventory_digest = _live_package(repo, commit)
    count = len(inventory)
    candidate = result["candidate"]
    expected_candidate = {
        "format": "s2-coverage-candidate-identity-v1",
        "repo": str(repo.resolve()),
        "commit": commit,
        "tree": tree,
        "clean": True,
        "status_porcelain_v1": "",
        "package_path": str(package),
        "package_python_file_count": count,
        "package_inventory_algorithm": "sha256-canonical-json-v1",
        "package_inventory_sha256": inventory_digest,
    }
    _required(candidate == expected_candidate, "Coverage candidate identity is stale")
    _required(
        _GIT_OBJECT_RE.fullmatch(commit) is not None
        and _GIT_OBJECT_RE.fullmatch(tree) is not None
        and git(repo, "rev-parse", "HEAD") == commit
        and git(repo, "rev-parse", "HEAD^{tree}") == tree
        and git(repo, "status", "--porcelain=v1", "--untracked-files=all") == "",
        "Live source candidate is not clean and exact",
    )
    candidate_path = _member(root, result["candidate_identity_artifact"], final_members)
    _required(
        result["candidate_identity_artifact"]["path"] in child["preparation"]
        and _strict_json(candidate_path) == candidate,
        "Candidate identity artifact differs",
    )

    binding_path = _member(root, result["package_binding_artifact"], final_members)
    binding = _strict_json(binding_path)
    labs = list(_PROTECTED_LABS)
    _required(
        [name for name in inventory if name.startswith("static/course/labs/")]
        == labs,
        "Protected lab source inventory differs",
    )
    _required(
        isinstance(binding, dict)
        and binding.get("format") == "s2-coverage-path-binding-v2"
        and binding.get("source_package") == str(package)
        and binding.get("expected_python_file_count") == count
        and binding.get("source_python_file_count") == count
        and binding.get("installed_python_file_count") == count
        and binding.get("original_installed_python_file_count") == count
        and binding.get("source_only") == []
        and binding.get("installed_only") == []
        and binding.get("mismatched") == []
        and binding.get("original_source_only") == []
        and binding.get("original_installed_only") == []
        and binding.get("original_mismatched") == []
        and binding.get("files") == inventory
        and binding.get("static_lab_python_files") == labs
        and binding.get("status") == "PASS"
        and result["package_binding_artifact"]["path"] in child["preparation"],
        "Coverage package-path binding differs",
    )
    for label in ("installed_package", "original_installed_package"):
        declared_root = Path(binding[label])
        _required(
            declared_root.is_absolute()
            and _directory_inventory(declared_root) == inventory,
            f"Coverage {label} bytes differ from the committed package",
        )

    wheel_raw = wheel.read_bytes()
    wheel_value = result["wheel"]
    _required(
        isinstance(wheel_value, dict)
        and set(wheel_value) == {"path", "size_bytes", "sha256"}
        and Path(wheel_value["path"]).name == wheel.name
        and wheel_value["size_bytes"] == len(wheel_raw)
        and wheel_value["sha256"] == sha256(wheel_raw),
        "Coverage wheel identity differs from the gate-built wheel",
    )
    preparation_ref = child["preparation"].get("preparation-result.json")
    _required(preparation_ref is not None, "Coverage preparation result is missing")
    preparation = _strict_json(_member(root, preparation_ref, final_members))
    _required(
        isinstance(preparation, dict)
        and preparation.get("format") == "s2-combined-coverage-preparation-v2"
        and preparation.get("status") == "PASS"
        and preparation.get("measurements_run") is False
        and preparation.get("runtime_original_stable") is True
        and preparation.get("legacy_runtime_original_stable") is True
        and preparation.get("canonical_labs_stable") is True
        and preparation.get("package_path_binding") == "PASS"
        and preparation.get("package_python_file_count") == count
        and preparation.get("protected_lab_python_file_count") == len(labs)
        and preparation.get("coverage_manifest_count") == 106
        and preparation.get("legacy_coverage_manifest_count") == 106
        and preparation.get("candidate") == candidate
        and preparation.get("wheel_sha256") == wheel_value["sha256"],
        "Coverage preparation result differs",
    )

    report_path = _member(root, result["coverage_report_artifact"], final_members)
    _required(report_path.name == "coverage.json", "Coverage report path differs")
    coverage = _strict_json(report_path)
    _required(
        isinstance(coverage, dict)
        and isinstance(coverage.get("files"), dict)
        and isinstance(coverage.get("totals"), dict)
        and canonical(coverage["totals"]) == canonical(result["totals"]),
        "Coverage report totals differ from the result",
    )
    combined_path = _member(root, result["combined_data"], final_members)
    _required(combined_path.name == ".coverage", "Combined coverage data path differs")
    combined_analysis, combined_raw = _combined_coverage_analysis(
        combined_path,
        package,
        inventory,
    )

    rows = result["files"]
    _required(isinstance(rows, list) and len(rows) == count, "Coverage file count differs")
    row_keys = {
        "path", "package_relative_path", "size_bytes", "sha256", "statements",
        "covered", "missing", "excluded", "percent", "missing_lines",
    }
    by_relative: dict[str, dict[str, object]] = {}
    totals = {"num_statements": 0, "covered_lines": 0, "missing_lines": 0, "excluded_lines": 0}
    expected_paths = {str((package / name).resolve()) for name in inventory}
    _required(set(coverage["files"]) == expected_paths, "Coverage source roster is incomplete")
    for row in rows:
        _required(isinstance(row, dict) and set(row) == row_keys, "Coverage file row differs")
        relative = row["package_relative_path"]
        _required(
            isinstance(relative, str)
            and relative in inventory
            and relative not in by_relative
            and row["path"] == str((package / relative).resolve())
            and row["size_bytes"] == inventory[relative]["size_bytes"]
            and row["sha256"] == inventory[relative]["sha256"],
            "Coverage file identity differs",
        )
        entry = coverage["files"][row["path"]]
        _required(isinstance(entry, dict) and isinstance(entry.get("summary"), dict), "Coverage file entry differs")
        summary = entry["summary"]
        analysis = combined_analysis[relative]
        statements = summary.get("num_statements")
        covered = summary.get("covered_lines")
        missing = summary.get("missing_lines")
        excluded = summary.get("excluded_lines")
        _required(
            all(_integer(value) and value >= 0 for value in (statements, covered, missing, excluded))
            and covered + missing == statements
            and excluded == 0
            and row["statements"] == statements
            and row["covered"] == covered
            and row["missing"] == missing
            and row["excluded"] == excluded
            and row["missing_lines"] == entry.get("missing_lines")
            and entry.get("excluded_lines") == []
            and isinstance(entry.get("executed_lines"), list)
            and entry["executed_lines"] == analysis["executed"]
            and entry["missing_lines"] == analysis["missing"]
            and entry["excluded_lines"] == analysis["excluded"]
            and statements == len(analysis["statements"])
            and len(set(entry["executed_lines"])) == covered
            and len(set(entry["missing_lines"])) == missing,
            "Coverage file counts or exclusions differ",
        )
        executed_set = set(entry["executed_lines"])
        missing_set = set(entry["missing_lines"])
        _required(
            all(
                _integer(line) and line > 0
                for line in entry["executed_lines"] + entry["missing_lines"]
            )
            and executed_set.isdisjoint(missing_set),
            "Coverage line roster differs",
        )
        expected_percent = 100.0 if statements == 0 else covered * 100.0 / statements
        _required(
            isinstance(row["percent"], (int, float))
            and not isinstance(row["percent"], bool)
            and math.isfinite(row["percent"])
            and math.isclose(float(row["percent"]), expected_percent, rel_tol=0.0, abs_tol=1e-9),
            "Coverage file percentage differs from integer counts",
        )
        _required(
            isinstance(summary.get("percent_covered"), (int, float))
            and not isinstance(summary.get("percent_covered"), bool)
            and math.isfinite(summary["percent_covered"])
            and math.isclose(
                float(summary["percent_covered"]),
                expected_percent,
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            "Coverage file percentage differs from integer counts",
        )
        by_relative[relative] = row
        totals["num_statements"] += statements
        totals["covered_lines"] += covered
        totals["missing_lines"] += missing
        totals["excluded_lines"] += excluded
    _required(list(by_relative) == sorted(by_relative), "Coverage rows are not sorted")
    for key, value in totals.items():
        _required(result["totals"].get(key) == value, f"Coverage total differs: {key}")
    statements = totals["num_statements"]
    covered = totals["covered_lines"]
    _required(
        statements > 0
        and result["threshold_percent"] == 95
        and covered * 100 >= statements * 95,
        "Complete-source coverage is below the exact 95 percent threshold",
    )
    expected_percent = covered * 100.0 / statements
    _required(
        math.isclose(
            float(result["totals"].get("percent_covered")),
            expected_percent,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "Coverage total percentage differs from integer counts",
    )

    _required(
        result["reported_python_file_count"] == count
        and result["expected_python_file_count"] == count
        and result["complete_inventory"]
        == {
            "status": "PASS",
            "expected_file_count": count,
            "reported_file_count": count,
            "package_inventory_algorithm": "sha256-canonical-json-v1",
            "package_inventory_sha256": inventory_digest,
        }
        and result["zero_exclusions"]
        == {"status": "PASS", "excluded_line_count": 0, "files_with_exclusions": []},
        "Coverage inventory or zero-exclusion proof differs",
    )
    expected_lab_rows = [by_relative[name] for name in labs]
    _required(result["protected_lab_files"] == expected_lab_rows, "Protected lab coverage roster differs")

    raw_rows = result["raw_data_files"]
    _required(
        isinstance(raw_rows, list)
        and result["raw_data_file_count"] == len(raw_rows)
        and {row.get("phase") for row in raw_rows if isinstance(row, dict)}
        == {"unit", "native", "legacy"},
        "Raw coverage roster differs",
    )
    _required(
        all(
            isinstance(row, dict)
            and set(row) == {"name", "size_bytes", "sha256", "phase"}
            for row in raw_rows
        )
        and len({row["name"] for row in raw_rows}) == len(raw_rows),
        "Raw coverage roster rows differ",
    )
    raw_directory = root / "raw"
    _required(
        raw_directory.is_dir()
        and not raw_directory.is_symlink()
        and {
            path.name
            for path in raw_directory.iterdir()
            if path.is_file() and path.name.startswith(".coverage")
        }
        == {row["name"] for row in raw_rows} | {".coverage"}
        and all(
            not path.is_symlink()
            for path in raw_directory.iterdir()
            if path.name.startswith(".coverage")
        ),
        "Coverage raw directory roster differs",
    )
    final_raw_members = {
        PurePosixPath(path).name: reference
        for path, reference in final_members.items()
        if path.startswith("raw/")
        and PurePosixPath(path).name.startswith(".coverage.")
    }
    _required(
        len(final_raw_members) == len(raw_rows)
        and set(final_raw_members) == {row["name"] for row in raw_rows}
        and all(
            final_raw_members[row["name"]]["size_bytes"] == row["size_bytes"]
            and final_raw_members[row["name"]]["sha256"] == row["sha256"]
            for row in raw_rows
        ),
        "Final evidence index raw coverage roster differs",
    )
    phases = result["phases"]
    _required(isinstance(phases, dict) and set(phases) == {"unit", "native", "legacy"}, "Coverage phases differ")
    qualifications = {
        "unit": "development_source_coverage_only",
        "native": "coverage_only_modified_runtime",
        "legacy": "coverage_only_modified_legacy_runtime",
    }
    cumulative_raw = 0
    raw_data_by_name: dict[str, dict[str, object]] = {}
    raw_union: dict[str, set[int]] = {
        relative: set()
        for relative in inventory
    }
    live_path_to_relative = {
        str((package / relative).resolve()): relative
        for relative in inventory
    }
    legacy_instrumented_ids: list[str] | None = None
    for phase in ("unit", "native", "legacy"):
        phase_row = phases[phase]
        phase_raw = [row for row in raw_rows if row["phase"] == phase]
        _required(
            isinstance(phase_row, dict)
            and phase_row.get("status") == "PASS"
            and phase_row.get("qualification") == qualifications[phase]
            and phase_row.get("exit_code") == 0
            and phase_row.get("command_exit_code") == 0
            and phase_row.get("source_stable") is True
            and phase_row.get("raw_data_file_count") == len(phase_raw)
            and phase_row.get("artifact_index") == indexes[phase],
            f"{phase} coverage phase differs",
        )
        phase_result_path = _member(root, phase_row.get("result_artifact"), final_members)
        _required(
            phase_row["result_artifact"]["path"] in child[phase],
            f"{phase} result is absent from its phase index",
        )
        phase_result = _strict_json(phase_result_path)
        cumulative_raw += len(phase_raw)
        _required(
            isinstance(phase_result, dict)
            and phase_result.get("format") == "s2-coverage-measurement-v3"
            and phase_result.get("phase") == phase
            and phase_result.get("qualification") == qualifications[phase]
            and phase_result.get("exit_code") == 0
            and phase_result.get("command_exit_code") == 0
            and phase_result.get("source_stable") is True
            and phase_result.get("raw_data_files_total") == cumulative_raw
            and phase_result.get("new_raw_files")
            == [{key: row[key] for key in ("name", "size_bytes", "sha256")} for row in phase_raw],
            f"{phase} measurement result differs",
        )
        instrumented = phase_row.get("instrumented")
        if phase == "unit":
            _required(
                instrumented is None
                and phase_result.get("instrumented_native") is None
                and phase_result.get("instrumented_legacy") is None,
                "Unit phase instrumented qualification differs",
            )
        elif phase == "native":
            native_keys = {
                "format",
                "qualification",
                "successful",
                "tests_run",
                "case_count",
                "supporting_test_count",
                "excluded_case_ids",
                "failures",
                "errors",
                "skipped",
                "expected_failures",
                "unexpected_successes",
                "artifact",
            }
            outcome_keys = (
                "failures",
                "errors",
                "skipped",
                "expected_failures",
                "unexpected_successes",
            )
            _required(
                isinstance(instrumented, dict)
                and set(instrumented) == native_keys
                and instrumented["format"] == "s2-native-instrumented-v2"
                and instrumented["qualification"] == qualifications[phase]
                and instrumented["successful"] is True
                and instrumented["tests_run"] == 54
                and instrumented["case_count"] == len(TEST_CASE_IDS)
                and instrumented["supporting_test_count"]
                == len(EXPECTED_SUPPORT_TESTS)
                and instrumented["excluded_case_ids"] == []
                and all(instrumented[key] == 0 for key in outcome_keys)
                and phase_result.get("instrumented_native") == instrumented
                and phase_result.get("instrumented_legacy") is None,
                "Instrumented native unittest summary differs",
            )
            instrumented_path = _member(
                root,
                instrumented["artifact"],
                final_members,
            )
            _required(
                instrumented["artifact"]["path"] in child[phase],
                "Instrumented native result is absent from its phase index",
            )
            instrumented_document = _strict_json(instrumented_path)
            document_keys = (native_keys - {"artifact"}) | {
                "started_at",
                "ended_at",
                "case_results",
                "supporting_test_results",
            }
            case_results = (
                instrumented_document.get("case_results", [])
                if isinstance(instrumented_document, dict)
                else []
            )
            supporting_results = (
                instrumented_document.get("supporting_test_results", [])
                if isinstance(instrumented_document, dict)
                else []
            )
            expected_case_ids = sorted(TEST_CASE_IDS)
            expected_scalar = {
                key: value
                for key, value in instrumented.items()
                if key != "artifact"
            }
            _required(
                isinstance(instrumented_document, dict)
                and set(instrumented_document) == document_keys
                and all(
                    instrumented_document.get(key) == value
                    for key, value in expected_scalar.items()
                )
                and isinstance(instrumented_document["started_at"], str)
                and bool(instrumented_document["started_at"])
                and isinstance(instrumented_document["ended_at"], str)
                and bool(instrumented_document["ended_at"])
                and isinstance(case_results, list)
                and len(case_results) == len(expected_case_ids)
                and [item.get("case_id") for item in case_results]
                == expected_case_ids
                and all(
                    isinstance(item, dict)
                    and set(item)
                    == {"case_id", "test", "status", "started_at", "ended_at"}
                    and item["status"] == "PASS"
                    and isinstance(item["test"], str)
                    and (
                        (match := NATIVE_CASE_RE.search(item["test"]))
                        is not None
                    )
                    and f"S2-NATIVE-{match.group('number')}" == item["case_id"]
                    for item in case_results
                )
                and isinstance(supporting_results, list)
                and [item.get("test") for item in supporting_results]
                == list(EXPECTED_SUPPORT_TESTS)
                and all(
                    isinstance(item, dict)
                    and set(item) == {"test", "status", "started_at", "ended_at"}
                    and item["status"] == "PASS"
                    for item in supporting_results
                )
                and len(
                    {
                        item["test"]
                        for item in [*case_results, *supporting_results]
                    }
                )
                == instrumented["tests_run"],
                "Instrumented native unittest result differs",
            )
        else:
            legacy_keys = {
                "format",
                "qualification",
                "successful",
                "tests_run",
                "test_ids",
                "stdout_artifact",
                "stderr_artifact",
            }
            _required(
                isinstance(instrumented, dict)
                and set(instrumented) == legacy_keys
                and instrumented["format"]
                == "s2-instrumented-protected-legacy-unittest-v1"
                and instrumented["qualification"] == qualifications[phase]
                and instrumented["successful"] is True
                and instrumented["tests_run"] == 7
                and isinstance(instrumented["test_ids"], list)
                and len(instrumented["test_ids"]) == 7
                and len(set(instrumented["test_ids"])) == 7
                and phase_result.get("instrumented_legacy") == instrumented
                and phase_result.get("instrumented_native") is None,
                "Instrumented protected legacy summary differs",
            )
            for stream in ("stdout_artifact", "stderr_artifact"):
                _member(root, instrumented[stream], final_members)
                _required(
                    instrumented[stream]["path"] in child[phase],
                    "Instrumented protected legacy stream is absent from its phase index",
                )
            legacy_instrumented_ids = instrumented["test_ids"]
        phase_index_raw = {
            PurePosixPath(path).name: reference
            for path, reference in child[phase].items()
            if path.startswith("raw/")
            and PurePosixPath(path).name.startswith(".coverage.")
        }
        _required(
            len(phase_index_raw) == len(phase_raw)
            and set(phase_index_raw) == {row["name"] for row in phase_raw}
            and all(
                phase_index_raw[row["name"]]["size_bytes"] == row["size_bytes"]
                and phase_index_raw[row["name"]]["sha256"] == row["sha256"]
                for row in phase_raw
            ),
            f"{phase} artifact index raw roster differs",
        )
        for raw_row in phase_raw:
            raw_path = _raw_member(root, raw_row, final_members, child[phase])
            raw_data = _coverage_data(raw_path)
            _required(
                raw_data["contexts"] == [phase],
                f"{phase} raw coverage context differs",
            )
            raw_data_by_name[raw_row["name"]] = raw_data
            if phase == "unit":
                _required(
                    set(raw_data["measured_files"]) <= set(live_path_to_relative),
                    "Unit raw coverage paths differ from the live source package",
                )
                for path, lines in raw_data["lines"].items():
                    raw_union[live_path_to_relative[path]].update(lines)

    roster_path = _member(root, result["raw_input_roster_artifact"], final_members)
    roster = _strict_json(roster_path)
    _required(
        isinstance(roster, dict)
        and roster
        == {
            "format": "s2-coverage-raw-input-roster-v1",
            "status": "PASS",
            "raw_data_file_count": len(raw_rows),
            "files": raw_rows,
            "unexpected_files": [],
            "missing_files": [],
        },
        "Raw coverage input roster differs",
    )
    native_authority = phases["native"].get("ordinary_authority")
    legacy_authority = phases["legacy"].get("ordinary_authority")
    _required(
        isinstance(native_authority, dict)
        and native_authority.get("artifact_path") == "ordinary-authority/native-result.json"
        and native_authority.get("aggregate") == "PASS"
        and native_authority.get("counts") == {"PASS": 25, "FAIL": 0, "BLOCKED": 0, "NOT_RUN": 0}
        and native_authority.get("supporting_tests") == len(EXPECTED_SUPPORT_TESTS)
        and native_authority.get("excluded_case_ids") == []
        and native_authority.get("wheel_sha256") == wheel_value["sha256"],
        "Ordinary native authority differs",
    )
    _required(
        isinstance(legacy_authority, dict)
        and legacy_authority.get("artifact_path") == "ordinary-authority/legacy-result.json"
        and legacy_authority.get("tests") == 7
        and legacy_authority.get("status") == "PASS"
        and legacy_authority.get("python") == "3.12.3"
        and legacy_authority.get("torch") == "2.11.0+cpu"
        and legacy_authority.get("product_qualification") == "NOT_RUN",
        "Ordinary legacy authority differs",
    )
    authority_documents: dict[str, object] = {}
    for phase, authority in (("native", native_authority), ("legacy", legacy_authority)):
        authority_ref = {
            "path": authority["artifact_path"],
            "size_bytes": authority["size_bytes"],
            "sha256": authority["sha256"],
        }
        authority_path = _member(root, authority_ref, final_members)
        _required(authority_ref["path"] in child[phase], f"{phase} authority is absent from its phase index")
        authority_documents[phase] = _strict_json(authority_path)
    native_document = authority_documents["native"]
    native_cases = native_document.get("cases", []) if isinstance(native_document, dict) else []
    native_supporting = (
        native_document.get("supporting_tests", [])
        if isinstance(native_document, dict)
        else []
    )
    _required(
        isinstance(native_document, dict)
        and native_document.get("format") == "llm-foundations-s2-native-development-v1"
        and native_document.get("denominator") == 25
        and isinstance(native_cases, list)
        and len(native_cases) == 25
        and [item.get("case_id") for item in native_cases]
        == [f"S2-NATIVE-{number:03d}" for number in range(1, 26)]
        and native_document.get("counts") == native_authority["counts"]
        and all(
            isinstance(item, dict) and item.get("status") == "PASS"
            for item in native_cases
        )
        and native_document.get("s2_native_development_gate") == "PASS"
        and native_document.get("unknown_or_duplicate_tests") == []
        and native_document.get("discovery_error") is None
        and isinstance(native_supporting, list)
        and [item.get("test") for item in native_supporting]
        == list(EXPECTED_SUPPORT_TESTS)
        and all(
            isinstance(item, dict) and item.get("status") == "PASS"
            for item in native_supporting
        )
        and native_document.get("development_exclusions")
        == {"enabled": False, "case_ids": [], "reason": None},
        "Copied ordinary native authority differs",
    )
    legacy_document = authority_documents["legacy"]
    legacy_tests = (
        legacy_document.get("tests", [])
        if isinstance(legacy_document, dict)
        else []
    )
    _required(
        isinstance(legacy_document, dict)
        and legacy_document.get("format") == "llm-foundations-s2-legacy-checks-v1"
        and legacy_document.get("status") == "PASS"
        and legacy_document.get("python") == legacy_authority["python"]
        and legacy_document.get("torch") == legacy_authority["torch"]
        and legacy_document.get("product_qualification") == "NOT_RUN"
        and isinstance(legacy_tests, list)
        and len(legacy_tests) == legacy_authority["tests"]
        and legacy_instrumented_ids
        == [item.get("test") for item in legacy_tests]
        and all(
            isinstance(item, dict) and item.get("status") == "PASS"
            for item in legacy_tests
        ),
        "Copied ordinary legacy authority differs",
    )

    native_summary = result["native_process_audit"]
    native_summary_keys = {
        "status",
        "expected_process_count",
        "observed_process_count",
        "expected_role_counts",
        "observed_role_counts",
        "complete_disjoint_role_roster",
        "job_roster_status",
        "expected_job_count",
        "observed_job_count",
        "expected_operation_counts",
        "observed_operation_counts",
        "artifact",
    }
    _required(
        isinstance(native_summary, dict)
        and set(native_summary) == native_summary_keys,
        "Native subprocess coverage summary differs",
    )
    native_audit_path = _member(root, native_summary["artifact"], final_members)
    native_audit = _strict_json(native_audit_path)
    native_raw = [row for row in raw_rows if row["phase"] == "native"]
    expected_process_count = sum(_EXPECTED_NATIVE_ROLE_COUNTS.values())
    expected_job_count = sum(_EXPECTED_NATIVE_OPERATION_COUNTS.values())
    expected_summary = {
        "status": "PASS",
        "expected_process_count": expected_process_count,
        "observed_process_count": expected_process_count,
        "expected_role_counts": _EXPECTED_NATIVE_ROLE_COUNTS,
        "observed_role_counts": _EXPECTED_NATIVE_ROLE_COUNTS,
        "complete_disjoint_role_roster": True,
        "job_roster_status": "PASS",
        "expected_job_count": expected_job_count,
        "observed_job_count": expected_job_count,
        "expected_operation_counts": _EXPECTED_NATIVE_OPERATION_COUNTS,
        "observed_operation_counts": _EXPECTED_NATIVE_OPERATION_COUNTS,
    }
    _required(
        all(native_summary.get(key) == value for key, value in expected_summary.items())
        and native_summary["artifact"]["path"] in child["native"],
        "Native subprocess coverage summary differs",
    )
    native_audit_keys = {
        "format",
        "status",
        "roots",
        "expected_process_count",
        "new_raw_file_count",
        "expected_role_counts",
        "observed_role_counts",
        "role_raw_files",
        "job_roster",
        "binding_failures",
        "contexts_ok",
        "pid_binding_ok",
        "distinct_parsed_pid_count",
        "complete_disjoint_role_roster",
        "files",
    }
    _required(
        isinstance(native_audit, dict)
        and set(native_audit) == native_audit_keys
        and native_audit.get("format") == "s2-native-raw-process-audit-v4"
        and native_audit.get("status") == "PASS"
        and native_audit.get("expected_process_count") == expected_process_count
        and native_audit.get("new_raw_file_count") == len(native_raw)
        and native_audit.get("expected_role_counts") == _EXPECTED_NATIVE_ROLE_COUNTS
        and native_audit.get("observed_role_counts") == _EXPECTED_NATIVE_ROLE_COUNTS
        and native_audit.get("binding_failures") == []
        and native_audit.get("contexts_ok") is True
        and native_audit.get("pid_binding_ok") is True
        and native_audit.get("distinct_parsed_pid_count") == expected_process_count
        and native_audit.get("complete_disjoint_role_roster") is True
        and isinstance(native_audit.get("files"), list),
        "Native subprocess coverage audit differs",
    )
    native_rows = {item.get("name"): item for item in native_audit["files"]}
    _required(
        len(native_rows) == len(native_audit["files"])
        and len(native_rows) == expected_process_count
        and set(native_rows) == {row["name"] for row in native_raw},
        "Native subprocess audit file roster differs",
    )
    native_roots_value = native_audit.get("roots")
    _required(
        isinstance(native_roots_value, dict)
        and set(native_roots_value)
        == {"source", "installed", "original_installed", "protected_labs"}
        and native_roots_value["source"] == str(package)
        and native_roots_value["installed"] == binding["installed_package"]
        and native_roots_value["original_installed"]
        == binding["original_installed_package"]
        and all(
            isinstance(value, str) and Path(value).is_absolute()
            for value in native_roots_value.values()
        ),
        "Native subprocess audited roots differ",
    )
    native_roots = {
        name: Path(value)
        for name, value in native_roots_value.items()
    }
    native_partitions = native_audit.get("role_raw_files")
    _required(
        isinstance(native_partitions, dict)
        and set(native_partitions) == set(_EXPECTED_NATIVE_ROLE_COUNTS)
        and all(
            isinstance(names, list)
            and names == sorted(set(names))
            and len(names) == _EXPECTED_NATIVE_ROLE_COUNTS[role]
            for role, names in native_partitions.items()
        )
        and set().union(*(set(names) for names in native_partitions.values()))
        == set(native_rows)
        and sum(len(names) for names in native_partitions.values())
        == len(native_rows),
        "Native subprocess role partitions differ",
    )
    observed_role_counts = {role: 0 for role in _EXPECTED_NATIVE_ROLE_COUNTS}
    for raw_row in native_raw:
        item = native_rows[raw_row["name"]]
        bindings = _audit_raw_row(
            item,
            raw_row,
            raw_data_by_name[raw_row["name"]],
            inventory,
            context="native",
            roles=set(_EXPECTED_NATIVE_ROLE_COUNTS),
            roots={"source", "installed", "original_installed"},
            root_paths=native_roots,
        )
        for relative, lines in bindings.items():
            raw_union[relative].update(lines)
        relative_paths = set(bindings)
        expected_role = (
            "service"
            if "service.py" in relative_paths
            else "worker"
            if "worker_main.py" in relative_paths
            else "preflight_support"
            if "preflight_main.py" in relative_paths
            else "cli_support"
            if "cli.py" in relative_paths
            else "runner_parent"
        )
        _required(
            item["role"] == expected_role
            and raw_row["name"] in native_partitions[expected_role],
            "Native subprocess role evidence differs",
        )
        observed_role_counts[expected_role] += 1
    _required(
        observed_role_counts == _EXPECTED_NATIVE_ROLE_COUNTS
        and native_audit["observed_role_counts"] == observed_role_counts
        and native_summary["observed_role_counts"] == observed_role_counts
        and len({item["pid"] for item in native_rows.values()})
        == expected_process_count
        and len({row["sha256"] for row in native_raw}) == len(native_raw),
        "Native subprocess raw identities or derived roles differ",
    )

    job_roster = native_audit.get("job_roster")
    _required(
        isinstance(job_roster, dict)
        and set(job_roster)
        == {
            "status",
            "expected_job_count",
            "observed_job_count",
            "expected_operation_counts",
            "observed_operation_counts",
            "jobs",
        }
        and job_roster.get("status") == "PASS"
        and job_roster.get("expected_job_count") == expected_job_count
        and job_roster.get("observed_job_count") == expected_job_count
        and job_roster.get("expected_operation_counts")
        == _EXPECTED_NATIVE_OPERATION_COUNTS
        and job_roster.get("observed_operation_counts")
        == _EXPECTED_NATIVE_OPERATION_COUNTS
        and isinstance(job_roster.get("jobs"), list)
        and len(job_roster["jobs"]) == expected_job_count,
        "Native job roster differs",
    )
    observed_operations = {key: 0 for key in _EXPECTED_NATIVE_OPERATION_COUNTS}
    observed_job_ids: list[str] = []
    observed_snapshot_paths: list[str] = []
    for job in job_roster["jobs"]:
        _required(
            isinstance(job, dict)
            and set(job) == {"job_id", "operation", "input_snapshot"}
            and isinstance(job["job_id"], str)
            and str(uuid.UUID(job["job_id"])) == job["job_id"]
            and job["operation"] in _EXPECTED_NATIVE_OPERATION_COUNTS,
            "Native job identity or operation differs",
        )
        snapshot_path = _member(root, job["input_snapshot"], final_members)
        snapshot_relative = job["input_snapshot"]["path"]
        _required(
            snapshot_relative in child["native"],
            "Native job input snapshot is absent from its phase index",
        )
        snapshot = _strict_json(snapshot_path)
        _required(
            isinstance(snapshot, dict)
            and snapshot.get("format") == "llm-foundations-worker-input-v1"
            and snapshot.get("job_id") == job["job_id"]
            and snapshot.get("operation") == job["operation"],
            "Native job input snapshot differs",
        )
        observed_job_ids.append(job["job_id"])
        observed_snapshot_paths.append(snapshot_relative)
        observed_operations[job["operation"]] += 1
    _required(
        observed_job_ids == sorted(set(observed_job_ids))
        and len(set(observed_snapshot_paths)) == expected_job_count
        and observed_operations == _EXPECTED_NATIVE_OPERATION_COUNTS
        and job_roster["observed_operation_counts"] == observed_operations
        and native_summary["observed_operation_counts"] == observed_operations
        and native_summary["observed_job_count"] == len(observed_job_ids),
        "Native job input snapshots do not establish the expected operation roster",
    )

    legacy_summary = result["legacy_path_audit"]
    _required(
        isinstance(legacy_summary, dict)
        and set(legacy_summary)
        == {
            "status",
            "parent_raw_file_count",
            "child_raw_file_count",
            "support_raw_file_count",
            "entrypoint_execution",
            "artifact",
        },
        "Protected legacy coverage summary differs",
    )
    legacy_audit_path = _member(root, legacy_summary["artifact"], final_members)
    legacy_audit = _strict_json(legacy_audit_path)
    legacy_raw = [row for row in raw_rows if row["phase"] == "legacy"]
    required_labs = [f"static/course/labs/{name}.py" for name in ("data", "lab", "model", "test_lab", "tokenizer")]
    _required(
        legacy_summary.get("status") == "PASS"
        and legacy_summary["artifact"]["path"] in child["legacy"]
        and isinstance(legacy_audit, dict)
        and legacy_audit.get("format") == "s2-legacy-raw-path-audit-v2"
        and legacy_audit.get("status") == "PASS"
        and legacy_audit.get("new_raw_file_count") == len(legacy_raw)
        and legacy_audit.get("contexts_ok") is True
        and legacy_audit.get("only_protected_execution") is True
        and legacy_audit.get("required_executed_files") == required_labs
        and legacy_audit.get("binding_failures") == []
        and isinstance(legacy_audit.get("files"), list),
        "Protected legacy coverage audit differs",
    )
    legacy_rows = {item.get("name"): item for item in legacy_audit["files"]}
    _required(
        len(legacy_rows) == len(legacy_audit["files"])
        and set(legacy_rows) == {row["name"] for row in legacy_raw},
        "Protected legacy audit file roster differs",
    )
    legacy_roots_value = legacy_audit.get("roots")
    _required(
        isinstance(legacy_roots_value, dict)
        and set(legacy_roots_value)
        == {"source", "installed", "original_installed", "protected_labs"}
        and legacy_roots_value["source"] == str(package)
        and legacy_roots_value["installed"] == binding["installed_package"]
        and legacy_roots_value["original_installed"]
        == binding["original_installed_package"]
        and native_roots_value == legacy_roots_value
        and all(
            isinstance(value, str) and Path(value).is_absolute()
            for value in legacy_roots_value.values()
        ),
        "Protected legacy audited roots differ",
    )
    legacy_roots = {
        name: Path(value)
        for name, value in legacy_roots_value.items()
    }
    legacy_partitions = {
        "unittest_parent": legacy_audit.get("parent_raw_files"),
        "lab_child": legacy_audit.get("child_raw_files"),
        "support": legacy_audit.get("support_raw_files"),
    }
    _required(
        all(
            isinstance(names, list)
            and len(names) == len(set(names))
            for names in legacy_partitions.values()
        )
        and set().union(*(set(names) for names in legacy_partitions.values()))
        == set(legacy_rows)
        and sum(len(names) for names in legacy_partitions.values())
        == len(legacy_rows)
        and legacy_audit.get("parent_raw_file_count")
        == len(legacy_partitions["unittest_parent"])
        and legacy_audit.get("child_raw_file_count")
        == len(legacy_partitions["lab_child"])
        and legacy_audit.get("support_raw_file_count")
        == len(legacy_partitions["support"])
        and len(legacy_rows) == 8
        and len(legacy_partitions["unittest_parent"]) == 1
        and len(legacy_partitions["lab_child"]) == 7
        and len(legacy_partitions["support"]) == 0,
        "Protected legacy process partitions differ",
    )
    _required(
        legacy_summary["parent_raw_file_count"]
        == len(legacy_partitions["unittest_parent"])
        and legacy_summary["child_raw_file_count"]
        == len(legacy_partitions["lab_child"])
        and legacy_summary["support_raw_file_count"]
        == len(legacy_partitions["support"]),
        "Protected legacy summary counts differ",
    )
    legacy_bindings: dict[str, dict[str, set[int]]] = {}
    legacy_pids: dict[str, int] = {}
    for raw_row in legacy_raw:
        item = legacy_rows[raw_row["name"]]
        bindings = _audit_raw_row(
            item,
            raw_row,
            raw_data_by_name[raw_row["name"]],
            inventory,
            context="legacy",
            roles={"unittest_parent", "lab_child", "support"},
            roots={"protected_labs"},
            root_paths=legacy_roots,
        )
        relative_paths = set(bindings)
        expected_role = (
            "unittest_parent"
            if "static/course/labs/test_lab.py" in relative_paths
            else "lab_child"
            if "static/course/labs/lab.py" in relative_paths
            else "support"
        )
        _required(
            item["role"] == expected_role
            and raw_row["name"] in legacy_partitions[expected_role],
            "Protected legacy process role evidence differs",
        )
        legacy_bindings[raw_row["name"]] = bindings
        legacy_pids[raw_row["name"]] = item["pid"]
        for relative, lines in bindings.items():
            raw_union[relative].update(lines)
    child_names = legacy_partitions["lab_child"]
    _required(
        len({legacy_pids[name] for name in child_names}) == len(child_names),
        "Protected legacy child process identities are not distinct",
    )
    _required(
        len(set(legacy_pids.values())) == len(legacy_pids)
        and len({row["sha256"] for row in legacy_raw}) == len(legacy_raw),
        "Protected legacy raw identities are not distinct",
    )
    entrypoint = legacy_audit.get("entrypoint_execution")
    main_children = [
        name
        for name in child_names
        if 184 in legacy_bindings[name].get("static/course/labs/lab.py", set())
    ]
    train_children = [
        name
        for name in child_names
        if 80 in legacy_bindings[name].get("static/course/labs/lab.py", set())
    ]
    _required(
        isinstance(entrypoint, dict)
        and set(entrypoint)
        == {
            "main_first_body_line",
            "train_first_body_line",
            "child_raw_files_executing_main",
            "child_raw_files_executing_train",
        }
        and entrypoint["main_first_body_line"] == 184
        and entrypoint["train_first_body_line"] == 80
        and sorted(entrypoint["child_raw_files_executing_main"])
        == sorted(main_children)
        and sorted(entrypoint["child_raw_files_executing_train"])
        == sorted(train_children)
        and set(main_children) == set(child_names)
        and len(train_children) >= 5,
        "Protected legacy child entrypoint execution differs",
    )
    _required(
        legacy_summary["entrypoint_execution"] == entrypoint,
        "Protected legacy entrypoint summary differs",
    )
    observed_legacy = sorted(
        set().union(*(set(bindings) for bindings in legacy_bindings.values()))
    )
    _required(
        legacy_audit.get("observed_executed_files") == observed_legacy
        and set(required_labs) <= set(observed_legacy),
        "Protected legacy executed-file union differs",
    )
    _required(
        combined_raw["contexts"] == ["legacy", "native", "unit"]
        and all(
            combined_raw["lines"][str((package / relative).resolve())]
            == sorted(raw_union[relative])
            for relative in inventory
        ),
        "Combined coverage database differs from the exact raw-data union",
    )

    scope = (
        "development coverage for the exact bound Python package inventory; "
        "ordinary locked-runtime and protected-legacy qualification remain separate"
    )
    _required(
        result["status"] == "PASS"
        and result["combine_exit_code"] == 0
        and result["report_exit_code"] == 0
        and result["json_exit_code"] == 0
        and result["original_runtime_unchanged"] is True
        and result["original_legacy_runtime_unchanged"] is True
        and result["canonical_labs_unchanged"] is True
        and result["source_stable"] is True
        and result["package_binding"] == "PASS"
        and result["scope"] == scope,
        "Coverage result did not establish the required bounded development claim",
    )
    return {
        "coverage_result": str(result_path),
        "package_python_file_count": count,
        "statements": statements,
        "covered": covered,
        "coverage_percent": expected_percent,
        "wheel_sha256": wheel_value["sha256"],
        "qualification": "development_source_coverage_only",
    }


def record_complete_source_coverage(
    gate: Gate,
    result_path: Path | None,
    *,
    repo: Path,
    commit: str,
    tree: str,
    wheel: Path | None,
) -> dict[str, object]:
    if wheel is None:
        return gate.blocked(
            "complete-source-coverage",
            "No uniquely identified gate-built wheel exists for coverage binding.",
        )
    if result_path is None:
        return gate.blocked(
            "complete-source-coverage",
            "A sealed complete-source coverage result was not provided.",
        )
    return gate.observe(
        "complete-source-coverage",
        lambda: verify_complete_source_coverage(
            result_path,
            repo=repo,
            commit=commit,
            tree=tree,
            wheel=wheel,
        ),
    )


def artifact_manifest(reports: Path) -> list[dict[str, object]]:
    output = []
    for path in sorted(reports.rglob("*")):
        if not path.is_file() or path.name == "result.json":
            continue
        raw = path.read_bytes()
        output.append(
            {
                "path": path.relative_to(reports).as_posix(),
                "size_bytes": len(raw),
                "sha256": sha256(raw),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--windows-python", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--canonical-course", type=Path, required=True)
    parser.add_argument("--legacy-python", type=Path, required=True)
    parser.add_argument(
        "--coverage-result",
        type=Path,
        help=(
            "Sealed result.json from the complete-source coverage campaign. "
            "Omission retains an explicit BLOCKED gate row."
        ),
    )
    parser.add_argument(
        "--allow-existing-companion",
        action="store_true",
        help=(
            "Permit a retry in a runtime containing a prior companion only after "
            "a complete installed RECORD audit."
        ),
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    reports = args.report_dir.resolve()
    if reports.is_relative_to(repo) or repo.is_relative_to(reports):
        parser.error("Keep retained S2 reports outside the source worktree")
    reports.mkdir(parents=True, exist_ok=False)
    runtime_python = launcher(args.runtime_python, parser, "--runtime-python")
    legacy_python = launcher(args.legacy_python, parser, "--legacy-python")
    gate = Gate(repo, reports)
    before = source_manifest(repo, reports)
    candidate_id = sha256(canonical(before))
    write_new(
        reports / "source-manifest.json",
        {"candidate_id": candidate_id, "files": before},
    )
    head = git(repo, "rev-parse", "HEAD")
    tree = git(repo, "rev-parse", "HEAD^{tree}")

    gate.observe("python-runtime", qualified_python)
    gate.observe("authoring-environment", lambda: authoring_environment(repo))
    gate.observe("s2-authority-before", lambda: s2_authority_audit(repo))
    gate.observe(
        "custody-before",
        lambda: custody_audit(repo, repo / "docs/implementation/s0/authority.json"),
    )
    gate.command(
        "runtime-lock-identities",
        [sys.executable, "-B", "scripts/build_runtime_identity.py", "--check"],
    )
    gate.command(
        "s0-regression-gate",
        [
            sys.executable,
            "-B",
            "scripts/run_s0_gate.py",
            "--report-dir",
            str(reports / "s0"),
            "--node",
            args.node,
            "--canonical-course",
            str(args.canonical_course),
            "--legacy-python",
            str(legacy_python),
        ],
        timeout=900,
    )
    gate.command(
        "bundled-contracts",
        [sys.executable, "-B", "scripts/build_runtime_contracts.py", "--check"],
    )
    gate.command(
        "bundled-reader",
        [sys.executable, "-B", "scripts/build_runtime_assets.py", "--check"],
    )
    unit = gate.command(
        "s2-wsl-unit-tests",
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "companion/tests",
            "-p",
            "no:cacheprovider",
            "-v",
            "--junitxml",
            str(reports / "unit-tests.xml"),
        ],
        timeout=900,
    )
    native_probe = repo / "companion/tests/native_windows_control_probe.py"
    native_path = subprocess.run(
        ["wslpath", "-w", str(native_probe)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    windows = gate.command(
        "native-windows-control",
        [args.windows_python, native_path],
        timeout=60,
    )
    if unit["status"] == "PASS":
        gate.observe(
            "unit-denominator",
            lambda: unit_denominator(
                reports / "unit-tests.xml", windows["status"] == "PASS"
            ),
        )
    else:
        gate.blocked("unit-denominator", "The WSL unit suite did not pass.")

    preinstall = gate.observe(
        "runtime-preinstall-state",
        lambda: runtime_preinstall(
            runtime_python,
            allow_existing_companion=args.allow_existing_companion,
        ),
    )
    legacy = gate.command(
        "legacy-lab-checks",
        [
            str(legacy_python),
            "-B",
            str(repo / "scripts/run_s2_legacy_checks.py"),
            "--canonical-course",
            str(args.canonical_course),
            "--report-dir",
            str(reports / "legacy"),
        ],
        timeout=300,
    )
    built = gate.command(
        "build-wheel",
        [
            sys.executable,
            "-B",
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(reports / "wheel"),
            "companion",
        ],
    )
    wheels = (
        list((reports / "wheel").glob("*.whl"))
        if built["status"] == "PASS"
        else []
    )
    wheel: Path | None = wheels[0] if len(wheels) == 1 else None
    if wheel is None:
        gate.blocked("wheel-source-binding", "The build did not create exactly one wheel.")
        record_complete_source_coverage(
            gate,
            args.coverage_result,
            repo=repo,
            commit=head,
            tree=tree,
            wheel=None,
        )
        gate.blocked("install-wheel", "No uniquely identified candidate wheel exists.")
        gate.blocked("runtime-environment", "The candidate wheel was not installed.")
        gate.blocked("installed-service-regressions", "The candidate wheel was not installed.")
        gate.blocked("native-model-checks", "The candidate wheel was not installed.")
        gate.blocked("native-denominator", "No native model result exists.")
    else:
        binding = gate.command(
            "wheel-source-binding",
            [
                sys.executable,
                "-B",
                "scripts/check_s1_wheel.py",
                str(wheel),
                str(repo),
                "--require-clean",
            ],
        )
        record_complete_source_coverage(
            gate,
            args.coverage_result,
            repo=repo,
            commit=head,
            tree=tree,
            wheel=wheel,
        )
        if binding["status"] == "PASS" and preinstall["status"] == "PASS":
            installed = gate.command(
                "install-wheel",
                [
                    str(runtime_python),
                    "-I",
                    "-m",
                    "pip",
                    "install",
                    "--force-reinstall",
                    "--no-index",
                    "--no-deps",
                    str(wheel),
                ],
                timeout=300,
            )
        else:
            installed = gate.blocked(
                "install-wheel",
                "Wheel binding or pristine-runtime verification did not pass.",
            )
        if installed["status"] == "PASS":
            runtime = gate.command(
                "runtime-environment",
                [
                    str(runtime_python),
                    "-I",
                    str(repo / "scripts/check_s1_runtime.py"),
                    str(repo / "companion/locks/wsl-cpu.wheels.json"),
                ],
            )
        else:
            runtime = gate.blocked(
                "runtime-environment", "The candidate wheel was not installed."
            )
        if runtime["status"] == "PASS":
            service = gate.command(
                "installed-service-regressions",
                [
                    sys.executable,
                    "-B",
                    "scripts/run_s1_installed_checks.py",
                    "--python",
                    str(runtime_python),
                    "--wheel",
                    str(wheel),
                    "--report-dir",
                    str(reports / "s1-installed"),
                ],
                timeout=600,
            )
        else:
            service = gate.blocked(
                "installed-service-regressions",
                "The locked installed runtime did not pass identity checks.",
            )
        if runtime["status"] == "PASS" and legacy["status"] == "PASS":
            native_argv = [
                str(runtime_python),
                "-I",
                str(repo / "scripts/run_s2_native_checks.py"),
                "--wheel",
                str(wheel),
                "--legacy-result",
                str(reports / "legacy/result.json"),
                "--report-dir",
                str(reports / "native"),
            ]
            native = gate.command(
                "native-model-checks",
                native_argv,
                timeout=1_800,
            )
        else:
            native = gate.blocked(
                "native-model-checks",
                "Runtime identity or the protected legacy suite did not pass.",
            )
        if native["status"] == "PASS":
            gate.observe(
                "native-denominator",
                lambda: native_denominator(reports / "native/result.json"),
            )
        else:
            gate.blocked(
                "native-denominator", "The native S2 development checks did not pass."
            )


    gate.observe("s2-authority-after", lambda: s2_authority_audit(repo))
    gate.observe(
        "custody-after",
        lambda: custody_audit(repo, repo / "docs/implementation/s0/authority.json"),
    )

    def stable() -> dict[str, object]:
        if source_manifest(repo, reports) != before:
            raise ValueError("Source candidate changed while S2 checks ran")
        if git(repo, "rev-parse", "HEAD") != head:
            raise ValueError("Source commit changed while S2 checks ran")
        return {"candidate_id": candidate_id, "git_commit": head, "git_tree": tree}

    gate.observe("candidate-stability", stable)
    seen = {item["gate_id"] for item in gate.rows}
    gate.rows.extend(
        {
            "gate_id": name,
            "status": "BLOCKED",
            "reason": "A required gate was not executed.",
        }
        for name in PLAN
        if name not in seen
    )
    complete = len(gate.rows) == len(PLAN)
    all_passed = (
        complete
        and all(item["status"] == "PASS" for item in gate.rows)
    )
    blocked = (
        complete
        and not any(item["status"] == "FAIL" for item in gate.rows)
        and any(item["status"] == "BLOCKED" for item in gate.rows)
    )
    aggregate = (
        "PASS"
        if all_passed
        else "BLOCKED"
        if blocked
        else "FAIL"
    )
    payload = {
        "format": "llm-foundations-s2-gate-v1",
        "s2_source_model_gate": aggregate,
        "candidate_id": candidate_id,
        "git_commit": head,
        "git_tree": tree,
        "gates": gate.rows,
        "artifacts": artifact_manifest(reports),
        "native_development_denominator": {
            "cases": 25,
            "required_passes": 25,
        },
        "canonical_acceptance": {
            "execution_units": 605,
            "passed": 0,
            "not_run": 605,
            "product_qualification": "NOT_RUN",
        },
        "mapped_s2_acceptance": {
            "execution_units": 20,
            "passed": 0,
            "not_run": 20,
        },
        "native_profile_qualification": {
            profile: "NOT_RUN"
            for profile in ("win-cpu", "win-cuda", "wsl-cpu", "wsl-cuda")
        },
    }
    write_new(reports / "result.json", payload)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
