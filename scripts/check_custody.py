"""Verify the frozen source and specification bytes without changing them."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
import sys

EXCLUSIONS = ["reviews/**", "spec-manifest.json", "**/__pycache__/**", "**/*.pyc"]


class CustodyError(ValueError):
    """The observed bytes do not match the declared authority."""


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def strict_json(path: Path) -> object:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise CustodyError(f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    def constant(value):
        raise CustodyError(f"Non-finite JSON number in {path}: {value}")

    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise CustodyError(f"Non-finite JSON number in {path}: {value}")
        return parsed

    try:
        return json.loads(path.read_bytes().decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=constant, parse_float=finite_float)
    except (OSError, UnicodeError, ValueError) as exc:
        raise CustodyError(f"Cannot read strict JSON {path}: {exc}") from exc


def contained(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise CustodyError(f"Invalid relative path: {relative!r}")
    parts = relative.split("/")
    if any(part in ("", ".", "..") for part in parts) or PurePosixPath(relative).is_absolute():
        raise CustodyError(f"Invalid relative path: {relative!r}")
    target = root.joinpath(*parts)
    for current in (target, *target.parents):
        if current == root:
            break
        if current.is_symlink():
            raise CustodyError(f"Symlink in custody path: {relative}")
    if not target.resolve().is_relative_to(root.resolve()):
        raise CustodyError(f"Path escapes custody root: {relative}")
    return target


def manifest_files(root: Path) -> list[dict]:
    rows = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if rel.parts[0] == "reviews" or "__pycache__" in rel.parts[:-1] or path.suffix == ".pyc":
            continue
        if rel.as_posix() == "spec-manifest.json":
            continue
        if path.is_symlink():
            raise CustodyError(f"Symlink in specification: {rel.as_posix()}")
        if path.is_file():
            raw = path.read_bytes()
            rows.append({"path": rel.as_posix(), "size_bytes": len(raw), "sha256": sha256(raw)})
    return sorted(rows, key=lambda row: row["path"])


def verify_spec(root: Path, expected: dict) -> dict:
    manifest_path = root / "spec-manifest.json"
    if sha256(manifest_path.read_bytes()) != expected["manifest_sha256"]:
        raise CustodyError(f"Specification manifest identity changed: {root}")
    manifest = strict_json(manifest_path)
    if manifest.get("exclusions") != EXCLUSIONS:
        raise CustodyError(f"Unexpected specification exclusions: {root}")
    if manifest.get("spec_revision") != expected["revision"]:
        raise CustodyError(f"Specification revision changed: {root}")
    rows = manifest["files"]
    seen = set()
    for row in rows:
        relative = row["path"]
        contained(root, relative)
        if relative in seen:
            raise CustodyError(f"Duplicate specification member: {relative}")
        seen.add(relative)
    observed = manifest_files(root)
    if observed != rows:
        actual = {row["path"]: row for row in observed}
        recorded = {row["path"]: row for row in rows}
        differences = sorted(key for key in actual.keys() | recorded.keys() if actual.get(key) != recorded.get(key))
        raise CustodyError(f"Specification members changed: {', '.join(differences[:12])}")
    payload = sha256(canonical(observed))
    if payload != expected["payload_sha256"] or payload != manifest.get("payload_sha256"):
        raise CustodyError(f"Specification payload identity changed: {root}")
    if len(rows) != expected["file_count"]:
        raise CustodyError(f"Specification file count changed: {root}")
    return {"revision": expected["revision"], "files": len(rows), "payload_sha256": payload,
            "manifest_sha256": expected["manifest_sha256"]}


def verify_protected(repo: Path, inventory_path: Path, expected: dict) -> dict:
    if sha256(inventory_path.read_bytes()) != expected["inventory_sha256"]:
        raise CustodyError("Protected-source receipt changed; a replacement receipt cannot redefine the baseline")
    inventory = strict_json(inventory_path)
    members = inventory["files_sha256"]
    if len(members) != expected["file_count"]:
        raise CustodyError("Protected-source file count changed")
    failures = []
    for relative, digest in sorted(members.items()):
        path = contained(repo, relative)
        if not path.is_file() or sha256(path.read_bytes()) != digest:
            failures.append(relative)
    if failures:
        raise CustodyError("Protected source changed or missing: " + ", ".join(failures))
    return {"files": len(members), "inventory_sha256": expected["inventory_sha256"]}


def verify_approved_history(repo: Path, expected: dict) -> dict:
    """Recheck the complete approved payload directly from its immutable Git commit."""
    commit = expected["commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise CustodyError("Approved authority requires a full Git commit ID")
    relative = expected["spec_root"]
    contained(repo, relative)
    stored = contained(repo, expected["retained_manifest"])
    if sha256(stored.read_bytes()) != expected["manifest_sha256"]:
        raise CustodyError("Retained revision 1.1 manifest identity changed")
    try:
        subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, "HEAD"],
                       check=True, capture_output=True, timeout=30)
        archived = subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", commit, "--", relative],
            check=True, capture_output=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise CustodyError(f"Approved baseline commit cannot be verified: {exc}") from exc
    rows = []
    original_manifest = None
    original_inventory = None
    with tarfile.open(fileobj=io.BytesIO(archived), mode="r:") as archive:
        for member in archive:
            if not member.name.startswith(relative + "/"):
                continue
            rel = member.name[len(relative) + 1:]
            parts = PurePosixPath(rel).parts
            if not parts or parts[0] == "reviews" or "__pycache__" in parts[:-1] or rel.endswith(".pyc"):
                continue
            if member.isdir():
                continue
            if not member.isfile():
                raise CustodyError(f"Non-regular approved specification member: {rel}")
            raw = archive.extractfile(member).read()
            if rel == "spec-manifest.json":
                original_manifest = raw
            else:
                if rel == "evidence/source-baseline.json":
                    original_inventory = raw
                rows.append({"path": rel, "size_bytes": len(raw), "sha256": sha256(raw)})
    if original_manifest is None or original_manifest != stored.read_bytes():
        raise CustodyError("Retained manifest differs from the approved Git object")
    manifest = strict_json(stored)
    rows.sort(key=lambda row: row["path"])
    payload = sha256(canonical(rows))
    if (rows != manifest["files"] or len(rows) != expected["file_count"]
            or payload != expected["payload_sha256"] or payload != manifest["payload_sha256"]
            or manifest["spec_revision"] != expected["revision"]
            or manifest["exclusions"] != EXCLUSIONS):
        raise CustodyError("Approved Git payload differs from its authority binding")
    if original_inventory is None:
        raise CustodyError("Approved Git payload has no protected-source inventory")
    inventory = json.loads(original_inventory)
    return {"revision": expected["revision"], "commit": commit, "files": len(rows),
            "payload_sha256": payload, "manifest_sha256": expected["manifest_sha256"],
            "protected_source": {"inventory_sha256": sha256(original_inventory),
                                 "file_count": len(inventory["files_sha256"])},
            "custody": "immutable Git objects plus retained manifest"}


def audit(repo_root: Path, authority_path: Path | None = None) -> dict:
    repo = repo_root.resolve()
    path = authority_path or repo / "docs/implementation/s0/authority.json"
    binding = strict_json(path)
    if binding.get("format") != "llm-foundations-s0-authority-v1":
        raise CustodyError("Unknown authority binding format")
    approved = binding["approved_baseline"]
    active = binding["active_specification"]
    active_root = contained(repo, binding["spec_root"])
    if active["spec_root"] != binding["spec_root"]:
        raise CustodyError("Active specification paths disagree")
    approved_result = verify_approved_history(repo, approved)
    specs = [approved_result, verify_spec(active_root, active)]
    protected_binding = binding["protected_source"]
    immutable_inventory = approved_result["protected_source"]
    if (protected_binding["inventory"] != binding["spec_root"] + "/evidence/source-baseline.json"
            or protected_binding["inventory_sha256"] != immutable_inventory["inventory_sha256"]
            or protected_binding["file_count"] != immutable_inventory["file_count"]):
        raise CustodyError("Protected-source binding differs from the approved Git inventory")
    protected = verify_protected(
        repo, contained(repo, protected_binding["inventory"]), immutable_inventory)
    return {"format": "llm-foundations-s0-custody-result-v1", "result": "PASS",
            "specifications": specs, "protected_source": protected,
            "product_qualification": "NOT_RUN"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--authority", type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit(args.repo_root, args.authority)
    except (CustodyError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"result": "FAIL", "error": str(exc), "product_qualification": "NOT_RUN"}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
