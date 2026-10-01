"""Run the complete S0 source gate, retaining every result without claiming qualification."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import importlib.metadata
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import xml.etree.ElementTree as ET

from check_custody import audit, canonical, contained, sha256, strict_json

GATE_PLAN = (
    "python-runtime", "custody-before", "spec-index", "contract-edges", "fixture-oracles",
    "fixture-materialization", "spec-seal", "contract-catalog", "dependency-locks", "authoring-environment", "s0-unit-tests",
    "unit-denominator", "legacy-runtime", "legacy-python-regressions", "legacy-regressions", "legacy-template-render", "legacy-source-audit",
    "custody-after", "candidate-stability",
)


def complete_results(rows: list[dict]) -> list[dict]:
    """Retain every mandatory S0 gate even when an earlier prerequisite fails."""
    seen = {row["gate_id"] for row in rows}
    return rows + [{"gate_id": name, "status": "BLOCKED",
                    "reason": "Not executed because a required setup or earlier gate failed"}
                   for name in GATE_PLAN if name not in seen]


def unit_denominator(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suites = list(root.iter("testsuite"))
    if not suites:
        raise ValueError("No unit-test suite was recorded")
    counts = {key: sum(int(suite.get(key, "0")) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    if counts["tests"] == 0 or any(counts[key] for key in ("failures", "errors", "skipped")):
        raise ValueError(f"Unit-test denominator incomplete: {counts}")
    return counts



def qualified_python() -> dict:
    if platform.python_implementation() != "CPython" or sys.version_info[:2] != (3, 12):
        raise ValueError("S0 requires CPython 3.12; other interpreters cannot qualify these checks")
    executable = Path(sys.executable).resolve()
    return {"implementation": platform.python_implementation(), "version": sys.version,
            "executable": str(executable), "executable_sha256": sha256(executable.read_bytes())}



def authoring_environment(repo: Path) -> dict:
    """Bind the validator process to every package in the development lock."""
    path = repo / "companion/locks/dev.requirements.txt"
    # The dependency-locks gate validates the complete lock grammar separately.
    pins = re.findall(r"^([A-Za-z0-9_.-]+)==([^ \\n]+)", path.read_text(encoding="utf-8"), re.MULTILINE)
    if not pins:
        raise ValueError("Development lock has no exact package pins")
    rows = []
    for name, version in pins:
        actual = importlib.metadata.version(name)
        if actual != version:
            raise ValueError(f"Authoring environment differs from lock: {name} expected {version}, found {actual}")
        rows.append({"name": name, "version": actual})
    return {"lock_sha256": sha256(path.read_bytes()), "packages": rows}



def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True, timeout=30).stdout.strip()


def source_manifest(repo: Path, reports: Path) -> list[dict]:
    raw = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        check=True, capture_output=True, timeout=30).stdout
    members = sorted(set(item.decode("utf-8") for item in raw.split(b"\0") if item))
    rows = []
    for name in members:
        path = contained(repo, name)
        if path.resolve().is_relative_to(reports.resolve()):
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative_parts = Path(name).parts
        if relative_parts[:2] == ("docs", "specs") and "reviews" in relative_parts[3:]:
            continue
        if not path.is_file():
            raise ValueError(f"Candidate input is missing: {name}")
        raw_bytes = path.read_bytes()
        rows.append({"path": name, "size_bytes": len(raw_bytes), "sha256": sha256(raw_bytes)})
    return rows


class Runner:
    def __init__(self, repo: Path, reports: Path):
        self.repo = repo
        self.reports = reports
        self.environment = os.environ.copy()
        self.environment["PYTHONDONTWRITEBYTECODE"] = "1"
        self.environment["PYTHONPATH"] = str(repo / "companion/src")
        self.environment["PYTHONUTF8"] = "1"
        self.results = []

    def command(self, name: str, argv: list[str], cwd: Path | None = None) -> dict:
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            result = subprocess.run(
                argv, cwd=cwd or self.repo, env=self.environment, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=120)
            stdout, stderr, returncode = result.stdout, result.stderr, result.returncode
            status = "PASS" if returncode == 0 else "FAIL"
            reason = None if returncode == 0 else f"Process exited {returncode}"
        except subprocess.TimeoutExpired as exc:
            def decoded(value):
                return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")
            stdout, stderr = decoded(exc.stdout), decoded(exc.stderr)
            returncode, status, reason = None, "BLOCKED", str(exc)
        except (OSError, subprocess.SubprocessError) as exc:
            stdout, stderr, returncode, status, reason = "", str(exc), None, "BLOCKED", str(exc)
        row = {"gate_id": name, "status": status, "command": argv, "cwd": str(cwd or self.repo),
               "started_at": started, "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "exit_code": returncode, "reason": reason}
        for label, value in (("stdout", stdout), ("stderr", stderr)):
            filename = f"{name}.{label}.txt"
            raw = value.encode("utf-8")
            (self.reports / filename).write_bytes(raw)
            row[label] = {"path": filename, "size_bytes": len(raw), "sha256": sha256(raw)}
        self.results.append(row)
        print(f"{name}: {status}", flush=True)
        return row

    def observation(self, name: str, action) -> dict:
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            value = action()
            row = {"gate_id": name, "status": "PASS", "observation": value, "reason": None}
        except Exception as exc:
            row = {"gate_id": name, "status": "FAIL", "observation": None, "reason": str(exc)}
        row.update(started_at=started, ended_at=dt.datetime.now(dt.timezone.utc).isoformat())
        self.results.append(row)
        print(f"{name}: {row['status']}", flush=True)
        return row


def legacy_audit(runner: Runner, node: str, legacy_python: str, canonical_course: Path, binding: dict) -> None:
    """Run the unchanged historical audit in a disposable copy because it writes a frozen file."""
    repo = runner.repo
    inventory = strict_json(contained(repo, binding["protected_source"]["inventory"]))
    hashes = strict_json(repo / "qa/source-hashes.json")
    with tempfile.TemporaryDirectory(prefix="llm-s0-legacy-") as temp:
        workspace = Path(temp)
        copy = workspace / "repo"
        original = workspace / "llm-foundations-v2"
        for relative in inventory["files_sha256"]:
            source = contained(repo, relative)
            target = copy / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for relative, digest in hashes.items():
            source = canonical_course / relative
            raw = source.read_bytes()
            if sha256(raw) != digest:
                raise ValueError(f"Canonical foundation source changed: {relative}")
            target = original / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        runtime_script = (
            "import importlib.metadata as m, json, sys, torch; "
            "assert sys.version_info[:2] == (3,12), 'Legacy regression environment requires Python 3.12'; "
            "assert torch.__version__ == '2.11.0+cpu', 'Legacy tests require the separate torch 2.11.0+cpu lock'; "
            "assert torch.version.cuda is None, 'Use the CPU-only legacy test environment'; "
            "print(json.dumps({'python':sys.version,'torch':torch.__version__,'packages':"
            "sorted((d.metadata['Name'],d.version) for d in m.distributions())},sort_keys=True))"
        )
        runtime = runner.command("legacy-runtime", [legacy_python, "-B", "-c", runtime_script], cwd=original)
        if runtime["status"] == "PASS":
            runner.command("legacy-python-regressions", [legacy_python, "-B", "-m", "unittest", "discover",
                                                         "-s", "labs", "-p", "test_lab.py", "-v"], cwd=original)
        runner.command("legacy-template-render", [node, "qa/templates.mjs"], cwd=copy)
        runner.command("legacy-source-audit", [sys.executable, "-B", "-X", "utf8", "qa/audit.py"], cwd=copy)
        generated = copy / "qa/accessibility-source.json"
        if generated.is_file():
            shutil.copyfile(generated, runner.reports / "legacy-accessibility-source.json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report-dir", type=Path, required=True,
                        help="New, empty result directory; an existing directory is never reused")
    parser.add_argument("--node", default="node", help="Node executable, including an absolute path if needed")
    parser.add_argument("--canonical-course", type=Path, required=True)
    parser.add_argument("--legacy-python", default=sys.executable,
                        help="Python in the separate pinned torch 2.11.0+cpu regression environment")
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve()
    reports = args.report_dir.resolve()
    if reports == repo or repo.is_relative_to(reports):
        parser.error("Report directory cannot contain the repository")
    if reports.is_relative_to(repo / "docs/specs"):
        parser.error("Use a result directory outside the sealed specification packages")
    reports.mkdir(parents=True, exist_ok=False)
    runner = Runner(repo, reports)
    try:
        return execute(args, repo, reports, runner)
    except Exception as exc:
        runner.results.append({"gate_id": "gate-execution", "status": "FAIL", "reason": str(exc)})
        write_new(reports / "result.json", {
            "format": "llm-foundations-s0-result-v1", "s0_gate": "FAIL",
            "gates": complete_results(runner.results), "product_qualification": "NOT_RUN",
            "reason": "Gate stopped unexpectedly; all already-written output is retained"})
        print(f"S0 FAIL: {exc}", file=sys.stderr)
        return 1


def execute(args, repo: Path, reports: Path, runner: Runner) -> int:
    authority_path = repo / "docs/implementation/s0/authority.json"
    python_runtime = runner.observation("python-runtime", qualified_python)
    if python_runtime["status"] != "PASS":
        write_new(reports / "result.json", {
            "format": "llm-foundations-s0-result-v1", "s0_gate": "FAIL",
            "gates": complete_results(runner.results), "product_qualification": "NOT_RUN"})
        return 1
    start = runner.observation("custody-before", lambda: audit(repo, authority_path))
    if start["status"] != "PASS":
        write_new(reports / "result.json", {
            "format": "llm-foundations-s0-result-v1", "s0_gate": "FAIL",
            "gates": complete_results(runner.results),
            "product_qualification": "NOT_RUN"})
        return 1
    binding = strict_json(authority_path)
    spec = contained(repo, binding["spec_root"])
    acceptance = strict_json(spec / "acceptance-cases.json")
    units = acceptance["execution_units"]
    if len(units) != binding["acceptance_execution_unit_count"] or any(row["status"] != "NOT_RUN" for row in units):
        raise ValueError("The frozen product denominator or its initial states changed")
    before = source_manifest(repo, reports)
    candidate_id = sha256(canonical(before))
    write_new(reports / "source-manifest.json", {
        "format": "llm-foundations-s0-source-manifest-v1", "candidate_id": candidate_id, "files": before})
    marker = "s0-" + uuid.uuid4().hex
    retained = []
    checks = [
        ("spec-index", "tools/check_spec.py", ["--check-index", "--report", f"reviews/{marker}-index.json"]),
        ("contract-edges", "tools/check_contract_edges.py", ["--strict", "--report", f"reviews/{marker}-edges.md"]),
        ("fixture-oracles", "tools/check_fixture_oracles.py", ["--report", f"reviews/{marker}-oracles.json"]),
        ("fixture-materialization", "fixtures/data/generate_fixtures.py", ["--check"]),
        ("spec-seal", "tools/seal_spec.py", ["--check"]),
    ]
    for name, script, arguments in checks:
        runner.command(name, [sys.executable, "-B", str(spec / script), *arguments])
        if "--report" in arguments:
            relative = arguments[arguments.index("--report") + 1]
            output = spec / relative
            if output.is_file():
                destination = f"{name}-report{output.suffix}"
                shutil.copyfile(output, reports / destination)
                retained.append({"source": output.relative_to(repo).as_posix(), "copy": destination,
                                 "sha256": sha256(output.read_bytes())})
    runner.command("contract-catalog", [sys.executable, "-B", "-m",
                                      "llm_foundations_companion.contracts", "--spec-root", str(spec)])
    runner.command("dependency-locks", [sys.executable, "-B", "companion/locks/validate.py"])
    runner.observation("authoring-environment", lambda: authoring_environment(repo))
    runner.command("s0-unit-tests", [sys.executable, "-B", "-m", "pytest", "companion/tests",
                                    "-p", "no:cacheprovider", "-v", "--strict-markers", "--strict-config",
                                    "--junitxml", str(reports / "unit-tests.xml")])
    runner.observation("unit-denominator", lambda: unit_denominator(reports / "unit-tests.xml"))
    runner.command("legacy-regressions", [args.node, "--test", "qa/core.test.mjs", "qa/import-regressions.test.mjs"])
    canonical_course = args.canonical_course.resolve()
    try:
        legacy_audit(runner, args.node, args.legacy_python, canonical_course, binding)
    except Exception as exc:
        runner.results.append({"gate_id": "legacy-source-audit-setup", "status": "BLOCKED", "reason": str(exc)})
    runner.observation("custody-after", lambda: audit(repo, authority_path))
    after = source_manifest(repo, reports)
    if before != after:
        before_map = {row["path"]: row for row in before}
        after_map = {row["path"]: row for row in after}
        differences = sorted(path for path in before_map.keys() | after_map.keys()
                             if before_map.get(path) != after_map.get(path))
        runner.results.append({"gate_id": "candidate-stability", "status": "FAIL",
                               "reason": "Candidate changed during the gate", "paths": differences})
    else:
        runner.results.append({"gate_id": "candidate-stability", "status": "PASS",
                               "candidate_id": candidate_id, "reason": None})
    artifact_rows = []
    for path in sorted(reports.iterdir()):
        if path.is_file():
            raw = path.read_bytes()
            artifact_rows.append({"path": path.name, "size_bytes": len(raw), "sha256": sha256(raw)})
    runner.results = complete_results(runner.results)
    status = "PASS" if all(row["status"] == "PASS" for row in runner.results) else "FAIL"
    result = {
        "format": "llm-foundations-s0-result-v1", "s0_gate": status,
        "candidate_id": candidate_id, "git_head": git(repo, "rev-parse", "HEAD"),
        "git_status": git(repo, "status", "--porcelain=v1", "--untracked-files=all"),
        "python": sys.version, "canonical_course": str(canonical_course),
        "legacy_python": args.legacy_python, "spec_revision": binding["active_specification"]["revision"],
        "spec_payload_sha256": binding["active_specification"]["payload_sha256"],
        "gates": runner.results, "retained_spec_reports": retained, "artifacts": artifact_rows,
        "product_qualification": "NOT_RUN",
        "acceptance_execution_units": {"total": len(units), "NOT_RUN": len(units)},
        "limitations": [
            "S0 verifies source custody, contracts, fixtures, the edition compiler, dependency metadata and regression scaffolding.",
            "No S1 service, complete applied curriculum, companion model job, profile install qualification or native browser acceptance is claimed.",
            "The original lab regression includes CPU tiny training/resume/evaluation/generation; it is legacy compatibility evidence.",
            "Specification-review GPU and dependency results remain historical evidence; this gate does not rerun them.",
        ],
    }
    write_new(reports / "result.json", result)
    print(json.dumps({"s0_gate": status, "candidate_id": candidate_id, "report": str(reports / "result.json"),
                      "product_qualification": "NOT_RUN"}))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
