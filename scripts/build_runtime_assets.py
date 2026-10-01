#!/usr/bin/env python3
"""Build/check the wheel-local public reader from the published ``dist`` tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Final


MANIFEST_FORMAT: Final = "llm-foundations-runtime-assets-v1"

# This is intentionally an explicit public-file decision. New files in ``dist``
# are rejected until they are reviewed and added here; no glob can copy spec,
# review, scratch, credential, or other repository material into the wheel.
PUBLISHED_ALLOWLIST: Final = (
    "app.js",
    "content.json",
    "core.js",
    "course.zip",
    "course/.gitignore",
    "course/REVIEW.md",
    "course/START_HERE.md",
    "course/VALIDATION.md",
    "course/curriculum.json",
    "course/design/CLAUDE_DESIGN_PROMPT.md",
    "course/design/INTERACTION_CONTRACT.md",
    "course/examples/bpe-check/manifest.json",
    "course/examples/bpe-check/metrics.jsonl",
    "course/examples/bpe-check/result.json",
    "course/examples/bpe-check/tokenizer.json",
    "course/examples/byte-training/manifest.json",
    "course/examples/byte-training/metrics.jsonl",
    "course/examples/byte-training/result.json",
    "course/examples/byte-training/sample.txt",
    "course/examples/byte-training/tokenizer.json",
    "course/examples/tokenizer-cases.json",
    "course/labs/WORKBOOK.md",
    "course/labs/data.py",
    "course/labs/lab.py",
    "course/labs/make_demo_data.py",
    "course/labs/model.py",
    "course/labs/test_lab.py",
    "course/labs/tokenizer.py",
    "course/lessons/00-setup.md",
    "course/lessons/01-prediction.md",
    "course/lessons/02-tokenizers.md",
    "course/lessons/03-data.md",
    "course/lessons/04-embeddings.md",
    "course/lessons/05-attention.md",
    "course/lessons/06-architecture.md",
    "course/lessons/07-training.md",
    "course/lessons/08-evaluation.md",
    "course/lessons/09-checkpoints.md",
    "course/lessons/10-experiments.md",
    "course/lessons/11-modern-models.md",
    "course/lessons/12-capstone.md",
    "course/original-source-manifest.json",
    "course/reference/GLOSSARY.md",
    "course/reference/SOURCES.md",
    "course/reference/TROUBLESHOOTING.md",
    "course/requirements.txt",
    "course/verification-summary.json",
    "index.html",
    "styles.css",
)

AUTHORED_ALLOWLIST: Final = (
    "local-index.html",
    "session.css",
    "session.js",
)


class BuildError(RuntimeError):
    """The published source or package copy violates the closed inventory."""


def _validate_allowlist(names: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    names = tuple(sorted(names))
    if len(names) != len(set(names)):
        raise BuildError(f"{label} asset allowlist contains a duplicate path")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise BuildError(f"unsafe asset allowlist path: {name!r}")
        if path.as_posix() != name or "\\" in name:
            raise BuildError(f"noncanonical asset allowlist path: {name!r}")
    return names


def _inventory(root: Path) -> tuple[set[str], list[str]]:
    files: set[str] = set()
    links: list[str] = []
    if not root.is_dir():
        return files, links
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            links.append(relative)
        elif path.is_file():
            files.add(relative)
    return files, sorted(links)


def _source_rows(
    source: Path,
    names: tuple[str, ...],
    *,
    source_prefix: str,
    label: str,
) -> tuple[dict[str, bytes], list[dict[str, object]]]:
    names = _validate_allowlist(names, label=label)
    actual, links = _inventory(source)
    expected = set(names)
    issues = [*(f"symbolic link {name}" for name in links)]
    issues.extend(f"missing {label} asset {name}" for name in sorted(expected - actual))
    issues.extend(f"unreviewed {label} asset {name}" for name in sorted(actual - expected))
    if issues:
        raise BuildError("; ".join(issues))

    outputs: dict[str, bytes] = {}
    rows: list[dict[str, object]] = []
    for name in names:
        raw = (source / PurePosixPath(name)).read_bytes()
        outputs[name] = raw
        rows.append(
            {
                "path": name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
                "source": f"{source_prefix}/{name}",
            }
        )
    return outputs, rows


def _expected_files(source: Path, authored_source: Path) -> dict[str, bytes]:
    outputs, published_rows = _source_rows(
        source,
        PUBLISHED_ALLOWLIST,
        source_prefix="dist",
        label="published",
    )
    authored, authored_rows = _source_rows(
        authored_source,
        AUTHORED_ALLOWLIST,
        source_prefix="scripts/runtime_assets",
        label="authored",
    )
    overlap = set(outputs) & set(authored)
    if overlap:
        raise BuildError("published/authored output collision: " + ", ".join(sorted(overlap)))
    outputs.update(authored)
    manifest = {
        "authored_files": authored_rows,
        "files": published_rows,
        "format": MANIFEST_FORMAT,
    }
    outputs["manifest.json"] = json.dumps(
        manifest,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
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
    actual, links = _inventory(output)
    issues = [*(f"unexpected symbolic link {name}" for name in links)]
    for name in sorted(expected):
        path = output / PurePosixPath(name)
        try:
            raw = path.read_bytes()
        except OSError:
            issues.append(f"missing {name}")
            continue
        if raw != expected[name]:
            issues.append(f"byte mismatch {name}")
    issues.extend(f"unexpected packaged asset {name}" for name in sorted(actual - set(expected)))
    return issues


def main(argv: list[str] | None = None) -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=repository / "dist")
    parser.add_argument(
        "--authored-source",
        type=Path,
        default=repository / "scripts" / "runtime_assets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repository
            / "companion"
            / "src"
            / "llm_foundations_companion"
            / "static"
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    source = args.source.resolve()
    authored_source = args.authored_source.resolve()
    output = args.output.resolve()
    try:
        expected = _expected_files(source, authored_source)
        if args.write:
            for name, raw in sorted(expected.items()):
                _write_atomic(output / PurePosixPath(name), raw)
        issues = _check(output, expected)
    except (BuildError, OSError) as exc:
        print(f"runtime assets: FAIL: {exc}", file=sys.stderr)
        return 1
    if issues:
        for issue in issues:
            print(f"runtime assets: FAIL: {issue}", file=sys.stderr)
        return 1
    print(
        "runtime assets: PASS "
        f"({len(PUBLISHED_ALLOWLIST)} published + "
        f"{len(AUTHORED_ALLOWLIST)} authored assets, byte exact)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
