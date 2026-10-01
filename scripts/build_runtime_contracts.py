#!/usr/bin/env python3
"""Build/check the wheel-local runtime contract bundle from the sealed spec."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


EXPECTED_SPEC_REVISION = "1.2"
MANIFEST_FORMAT = "llm-foundations-runtime-contracts-v1"


class BuildError(RuntimeError):
    pass


def _strict_json(path: Path) -> Any:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> Any:
        raise ValueError(f"non-finite number {token}")

    try:
        raw = path.read_bytes()
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BuildError(f"cannot read strict JSON {path}: {exc}") from exc


def _source_inventory(spec_root: Path) -> tuple[dict[str, dict[str, Any]], str]:
    manifest_path = spec_root / "spec-manifest.json"
    manifest = _strict_json(manifest_path)
    if not isinstance(manifest, dict):
        raise BuildError("sealed spec manifest must be an object")
    if manifest.get("format") != "llm-foundations-spec-manifest-v1":
        raise BuildError("unexpected sealed spec manifest format")
    if manifest.get("spec_revision") != EXPECTED_SPEC_REVISION:
        raise BuildError(
            f"expected sealed spec revision {EXPECTED_SPEC_REVISION}, got "
            f"{manifest.get('spec_revision')!r}"
        )
    payload = manifest.get("payload_sha256")
    if not isinstance(payload, str) or len(payload) != 64:
        raise BuildError("sealed spec manifest has no valid payload digest")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise BuildError("sealed spec manifest files must be an array")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise BuildError("invalid sealed spec manifest row")
        source = row["path"]
        if source in indexed:
            raise BuildError(f"duplicate sealed spec manifest path: {source}")
        indexed[source] = row
    return indexed, payload


def _selected_sources(spec_root: Path) -> list[tuple[str, Path]]:
    contracts = spec_root / "contracts"
    paths = sorted(contracts.glob("*.json"), key=lambda path: path.name)
    paths.extend(sorted((contracts / "schemas").glob("*.json"), key=lambda path: path.name))
    selected = [
        (path.relative_to(contracts).as_posix(), path)
        for path in paths
    ]
    selected.append(
        (
            "materialized-manifest.json",
            spec_root
            / "fixtures"
            / "data"
            / "materialized"
            / "materialized-manifest.json",
        )
    )
    if not selected:
        raise BuildError("no runtime contract JSON sources found")
    return selected


def _expected_files(spec_root: Path) -> dict[str, bytes]:
    source_rows, payload = _source_inventory(spec_root)
    outputs: dict[str, bytes] = {}
    manifest_rows: list[dict[str, Any]] = []
    for name, source_path in _selected_sources(spec_root):
        source = source_path.relative_to(spec_root).as_posix()
        row = source_rows.get(source)
        if row is None:
            raise BuildError(f"runtime contract is absent from sealed manifest: {source}")
        raw = source_path.read_bytes()
        size = len(raw)
        digest = hashlib.sha256(raw).hexdigest()
        if row.get("size_bytes") != size or row.get("sha256") != digest:
            raise BuildError(f"sealed runtime contract hash mismatch: {source}")
        _strict_json(source_path)
        if name in outputs:
            raise BuildError(f"duplicate runtime contract output name: {name}")
        outputs[name] = raw
        manifest_rows.append(
            {
                "name": name,
                "sha256": digest,
                "size_bytes": size,
                "source": source,
            }
        )
    bundle_manifest = {
        "files": manifest_rows,
        "format": MANIFEST_FORMAT,
        "spec_payload_sha256": payload,
        "spec_revision": EXPECTED_SPEC_REVISION,
    }
    outputs["manifest.json"] = (
        json.dumps(bundle_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return outputs


def _write_atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(raw)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _check(output: Path, expected: dict[str, bytes]) -> list[str]:
    issues: list[str] = []
    expected_names = set(expected)
    actual_names = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*.json")
        if path.is_file()
    } if output.is_dir() else set()
    for name in sorted(expected_names):
        path = output / name
        try:
            actual = path.read_bytes()
        except OSError:
            issues.append(f"missing {name}")
            continue
        if actual != expected[name]:
            issues.append(f"byte mismatch {name}")
    for name in sorted(actual_names - expected_names):
        issues.append(f"unexpected JSON {name}")
    return issues


def main(argv: list[str] | None = None) -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec-root",
        type=Path,
        default=repository / "docs" / "specs" / "intermediate-v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repository
            / "companion"
            / "src"
            / "llm_foundations_companion"
            / "contract_data"
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    spec_root = args.spec_root.resolve()
    output = args.output.resolve()
    try:
        expected = _expected_files(spec_root)
        if args.write:
            for name, raw in sorted(expected.items()):
                _write_atomic(output / name, raw)
        issues = _check(output, expected)
    except (BuildError, OSError) as exc:
        print(f"runtime contracts: FAIL: {exc}", file=sys.stderr)
        return 1
    if issues:
        for issue in issues:
            print(f"runtime contracts: FAIL: {issue}", file=sys.stderr)
        return 1
    print(
        f"runtime contracts: PASS ({len(expected) - 1} sealed documents, "
        f"spec revision {EXPECTED_SPEC_REVISION})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
