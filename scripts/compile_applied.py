#!/usr/bin/env python3
"""Compile the versioned intermediate curriculum overlay.

The compiler intentionally uses only the Python standard library. It treats
the protected foundation curriculum, the active specification fixtures, and
the applied curriculum sources as inputs. Validation finishes before an
output file is created or replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit


SOURCE_FORMAT = "llm-foundations-applied-source-v1"
ANSWERS_FORMAT = "llm-foundations-applied-answers-v1"
RUBRICS_FORMAT = "llm-foundations-applied-rubrics-v1"
OUTPUT_FORMAT = "llm-foundations-applied-content-v1"
OVERLAY_FORMAT = "llm-foundations-curriculum-overlay-v1"
ORACLES_FORMAT = "llm-foundations-exercise-oracles-v1"
FOUNDATION_IDS = tuple(f"{value:02d}" for value in range(13))
APPLIED_IDS = ("P00",) + tuple(str(value) for value in range(13, 21)) + ("E01", "E02")
MODULE_ORDER = ("P00",) + tuple(f"{value:02d}" for value in range(21)) + ("E01", "E02")
LESSON_SECTIONS = (
    "Outcome and boundary",
    "Predict before running",
    "Guided worked example",
    "Partially scaffolded practice",
    "Independent evidence task",
    "Feedback and limits",
    "Stopping point",
)
CARD_FIELDS = (
    "card_id",
    "source_lesson_id",
    "anchor_heading",
    "anchor_occurrence",
    "order",
    "kind",
    "destination_module_id",
    "title",
    "summary",
    "estimated_minutes",
    "prerequisites",
)
CARD_KINDS = frozenset(("prerequisite", "required", "elective"))
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
EXPLICIT_ANCHOR_RE = re.compile(r"[ \t]+\{#([A-Za-z0-9][A-Za-z0-9._:-]*)\}[ \t]*$")
EXERCISE_MARKER_RE = re.compile(
    r"<!--[ \t]*exercise:([A-Za-z0-9][A-Za-z0-9._-]*)[ \t]*-->", re.IGNORECASE
)
LINK_RE = re.compile(
    r"!?\[[^\]\n]*\]\([ \t]*(?:<([^>\n]+)>|([^\s)]+))(?:[ \t]+[\"'][^\n)]*[\"'])?[ \t]*\)"
)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
REFERENCE_DEFINITION_RE = re.compile(r"^[ \t]{0,3}\[[^\]\n]+\]:", re.MULTILINE)
REFERENCE_USE_RE = re.compile(r"!?\[[^\]\n]+\][ \t]*\[[^\]\n]*\]")
AUTOLINK_RE = re.compile(
    r"<(?:[A-Za-z][A-Za-z0-9+.-]{1,31}:[^<>\s]*|[^<>\s@]+@[^<>\s@]+)>",
    re.IGNORECASE,
)
HTML_LINK_ATTRIBUTE_RE = re.compile(
    r"<[A-Za-z][^>]*\s(?:href|src)\s*=", re.IGNORECASE
)
PLACEHOLDER_RE = re.compile(
    r"^(?:todo|tbd|placeholder|coming soon|under construction)[.!]?$", re.IGNORECASE
)


class CompilationError(ValueError):
    """A deterministic source or contract validation failure."""


def _fail(message: str) -> None:
    raise CompilationError(message)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_json(path: Path, label: str) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        _fail(f"{label}: cannot read strict UTF-8 JSON: {exc}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{label}: duplicate object key {key!r}")
            result[key] = value
        return result

    def finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            _fail(f"{label}: non-finite number {token}")
        return value

    try:
        return json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: _fail(f"{label}: non-finite number {token}"),
            parse_float=finite_float,
        )
    except json.JSONDecodeError as exc:
        _fail(f"{label}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}")


def _require_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{where}: expected an object")
    return value


def _require_list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{where}: expected an array")
    return value


def _require_fields(
    value: Any,
    required: Iterable[str],
    allowed: Iterable[str],
    where: str,
) -> dict[str, Any]:
    obj = _require_object(value, where)
    required_set = set(required)
    allowed_set = set(allowed)
    missing = sorted(required_set - obj.keys())
    unknown = sorted(obj.keys() - allowed_set)
    if missing:
        _fail(f"{where}: missing fields: {', '.join(missing)}")
    if unknown:
        _fail(f"{where}: unknown fields: {', '.join(unknown)}")
    return obj


def _require_string(value: Any, where: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        _fail(f"{where}: expected a string")
    if nonempty and not value.strip():
        _fail(f"{where}: expected a nonempty string")
    return value


def _require_id(value: Any, where: str) -> str:
    identifier = _require_string(value, where)
    if not ID_RE.fullmatch(identifier):
        _fail(f"{where}: invalid identifier {identifier!r}")
    return identifier


def _unique_index(items: list[Any], key: str, where: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(items):
        obj = _require_object(raw, f"{where}[{position}]")
        identifier = _require_id(obj.get(key), f"{where}[{position}].{key}")
        if identifier in indexed:
            _fail(f"{where}: duplicate {key} {identifier!r}")
        indexed[identifier] = obj
    return indexed


def _safe_relative_file(root: Path, relative: Any, where: str, *, suffix: str | None = None) -> Path:
    text = _require_string(relative, where)
    if "\\" in text:
        _fail(f"{where}: use POSIX separators")
    pure = PurePosixPath(text)
    if pure.is_absolute() or not pure.parts or any(part in ("", ".", "..") for part in pure.parts):
        _fail(f"{where}: expected a normalized relative path")
    if suffix is not None and pure.suffix != suffix:
        _fail(f"{where}: expected a {suffix} file")
    candidate = root.joinpath(*pure.parts)
    if candidate.is_symlink():
        _fail(f"{where}: symlink sources are not allowed")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        _fail(f"{where}: source does not exist: {exc}")
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError:
        _fail(f"{where}: path escapes its source root")
    if not resolved.is_file():
        _fail(f"{where}: expected a regular file")
    return resolved


def _read_utf8(path: Path, where: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        _fail(f"{where}: cannot read strict UTF-8 text: {exc}")


def _strip_explicit_anchor(title: str) -> tuple[str, str | None]:
    match = EXPLICIT_ANCHOR_RE.search(title)
    if match is None:
        return title.strip(), None
    return title[: match.start()].strip(), match.group(1)


def _slug(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    chars: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if char in ("-", "_") or category[0] in ("L", "N"):
            chars.append(char)
        elif char.isspace():
            chars.append("-")
    return re.sub(r"-+", "-", "".join(chars)).strip("-")


def _headings(markdown: str, where: str) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    visible = _without_fenced_code(markdown)
    for match in HEADING_RE.finditer(visible):
        title, explicit = _strip_explicit_anchor(match.group(2))
        base = explicit or _slug(title)
        if not base:
            _fail(f"{where}: heading at line {markdown.count(chr(10), 0, match.start()) + 1} has no anchor")
        count = counts.get(base, 0)
        anchor = base if count == 0 else f"{base}-{count}"
        counts[base] = count + 1
        headings.append(
            {
                "level": len(match.group(1)),
                "title": title,
                "anchor": anchor,
                "start": match.start(),
                "end": match.end(),
                "line": markdown.count("\n", 0, match.start()) + 1,
            }
        )
    return headings


def _without_fenced_code(markdown: str) -> str:
    def masked(line: str) -> str:
        return "".join(char if char in "\r\n" else " " for char in line)

    result: list[str] = []
    fence: str | None = None
    for line in markdown.splitlines(keepends=True):
        stripped = line.lstrip()
        marker = stripped[:3]
        if fence is None and marker in ("```", "~~~"):
            fence = marker
            result.append(masked(line))
        elif fence is not None and stripped.startswith(fence):
            fence = None
            result.append(masked(line))
        elif fence is None:
            result.append(line)
        else:
            result.append(masked(line))
    return "".join(result)


def _visible_section_text(markdown: str) -> str:
    visible = HTML_COMMENT_RE.sub(" ", markdown)
    visible = re.sub(r"^[ \t]*(?:```|~~~).*?$", " ", visible, flags=re.MULTILINE)
    visible = re.sub(r"\]\([^\n)]*\)", "]", visible)
    visible = re.sub(r"<[^>\n]+>", " ", visible)
    visible = re.sub(r"[*_~`]", "", visible)
    visible = re.sub(r"^[ \t]*#{1,6}[ \t]+", "", visible, flags=re.MULTILINE)
    visible = re.sub(r"[!\[\]]", "", visible)
    visible = re.sub(r"\s+", " ", visible)
    return visible.strip()


def _validate_applied_markdown(
    module_id: str,
    title: str,
    markdown: str,
    declared_exercises: list[str],
    where: str,
) -> set[str]:
    headings = _headings(markdown, where)
    h1 = [heading for heading in headings if heading["level"] == 1]
    if len(h1) != 1:
        _fail(f"{where}: expected exactly one level-1 heading")
    expected_h1 = f"{module_id} — {title}"
    if h1[0]["title"] != expected_h1:
        _fail(f"{where}: level-1 heading must be {expected_h1!r}")

    h2 = [heading for heading in headings if heading["level"] == 2]
    actual_sections = [heading["title"] for heading in h2]
    if actual_sections != list(LESSON_SECTIONS):
        _fail(
            f"{where}: level-2 sections must be exactly, in order: "
            + ", ".join(LESSON_SECTIONS)
        )
    for index, heading in enumerate(h2):
        end = h2[index + 1]["start"] if index + 1 < len(h2) else len(markdown)
        content = markdown[heading["end"] : end]
        visible = _visible_section_text(content)
        if not re.search(r"\w", visible, re.UNICODE):
            _fail(f"{where}: section {heading['title']!r} is empty")
        if PLACEHOLDER_RE.fullmatch(visible):
            _fail(f"{where}: section {heading['title']!r} is a placeholder")

    markers = EXERCISE_MARKER_RE.findall(_without_fenced_code(markdown))
    duplicates = sorted({identifier for identifier in markers if markers.count(identifier) > 1})
    if duplicates:
        _fail(f"{where}: duplicate exercise markers: {', '.join(duplicates)}")
    if set(markers) != set(declared_exercises):
        missing = sorted(set(declared_exercises) - set(markers))
        unknown = sorted(set(markers) - set(declared_exercises))
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        _fail(f"{where}: exercise marker mismatch ({'; '.join(details)})")
    return {heading["anchor"] for heading in headings}


def _validate_foundation_curriculum(course_root: Path) -> tuple[dict[str, dict[str, Any]], Path]:
    curriculum_path = course_root / "curriculum.json"
    curriculum = _require_object(_load_json(curriculum_path, "foundation curriculum"), "foundation curriculum")
    modules = _require_list(curriculum.get("modules"), "foundation curriculum.modules")
    index = _unique_index(modules, "id", "foundation curriculum.modules")
    if tuple(index) != FOUNDATION_IDS:
        _fail("foundation curriculum: module IDs/order must be exactly 00 through 12")
    for module_id, module in index.items():
        if module.get("sequence") != int(module_id):
            _fail(f"foundation module {module_id}: sequence mismatch")
        _require_string(module.get("title"), f"foundation module {module_id}.title")
        _require_string(module.get("outcome"), f"foundation module {module_id}.outcome")
        module["_source_path"] = _safe_relative_file(
            course_root, module.get("path"), f"foundation module {module_id}.path", suffix=".md"
        )
    return index, curriculum_path


def _validate_cards(
    cards_value: Any,
    foundation_bodies: dict[str, str],
    module_ids: set[str],
) -> list[dict[str, Any]]:
    cards = _require_list(cards_value, "curriculum overlay.cards")
    _unique_index(cards, "card_id", "curriculum overlay.cards")
    destinations: list[str] = []
    insertion_keys: set[tuple[str, str, int, int]] = set()
    for position, raw in enumerate(cards):
        card = _require_fields(
            raw,
            CARD_FIELDS,
            CARD_FIELDS,
            f"curriculum overlay.cards[{position}]",
        )
        source_id = _require_string(card["source_lesson_id"], f"card {position}.source_lesson_id")
        destination_id = _require_string(
            card["destination_module_id"], f"card {position}.destination_module_id"
        )
        if source_id not in foundation_bodies:
            _fail(f"card {position}: unknown foundation source lesson {source_id!r}")
        if destination_id not in module_ids:
            _fail(f"card {position}: unknown destination module {destination_id!r}")
        if destination_id in destinations:
            _fail(f"card {position}: duplicate destination module {destination_id!r}")
        anchor_heading = _require_string(card["anchor_heading"], f"card {position}.anchor_heading")
        occurrence = card["anchor_occurrence"]
        order = card["order"]
        if occurrence != 1 or isinstance(occurrence, bool):
            _fail(f"card {position}.anchor_occurrence: expected exactly 1")
        if not isinstance(order, int) or isinstance(order, bool) or order < 0:
            _fail(f"card {position}.order: expected a nonnegative integer")
        kind = _require_string(card["kind"], f"card {position}.kind")
        if kind not in CARD_KINDS:
            _fail(f"card {position}.kind: expected prerequisite, required, or elective")
        expected_kind = (
            "prerequisite"
            if destination_id == "P00"
            else "elective"
            if destination_id in ("E01", "E02")
            else "required"
        )
        if kind != expected_kind:
            _fail(f"card {position}.kind: {kind!r} does not match destination {destination_id}")
        _require_string(card["title"], f"card {position}.title")
        _require_string(card["summary"], f"card {position}.summary")
        minutes = card["estimated_minutes"]
        if not isinstance(minutes, int) or isinstance(minutes, bool) or minutes <= 0:
            _fail(f"card {position}.estimated_minutes: expected a positive integer")
        prerequisites = _require_list(card["prerequisites"], f"card {position}.prerequisites")
        normalized_prerequisites: list[str] = []
        for prerequisite_position, prerequisite in enumerate(prerequisites):
            prerequisite_id = _require_string(
                prerequisite, f"card {position}.prerequisites[{prerequisite_position}]"
            )
            if prerequisite_id not in module_ids:
                _fail(f"card {position}: unknown prerequisite module {prerequisite_id!r}")
            if prerequisite_id in normalized_prerequisites:
                _fail(f"card {position}: duplicate prerequisite module {prerequisite_id!r}")
            normalized_prerequisites.append(prerequisite_id)
        matching = [
            heading
            for heading in _headings(foundation_bodies[source_id], f"foundation lesson {source_id}")
            if ("#" * heading["level"] + " " + heading["title"]) == anchor_heading
        ]
        if len(matching) != 1:
            _fail(f"card {position}: anchor {anchor_heading!r} is absent or ambiguous in lesson {source_id}")
        insertion_key = (source_id, anchor_heading, occurrence, order)
        if insertion_key in insertion_keys:
            _fail(f"card {position}: duplicate supplement-card insertion point/order")
        insertion_keys.add(insertion_key)
        destinations.append(destination_id)
    if sorted(destinations) != sorted(APPLIED_IDS):
        _fail("curriculum overlay.cards: destinations must cover each applied module exactly once")
    return cards


def _validate_link_target(
    target: str,
    source_path: Path,
    source_where: str,
    module_paths: dict[str, Path],
    anchors_by_path: dict[Path, set[str]],
    repo_root: Path,
) -> None:
    target = unquote(target.strip())
    if target.startswith("module:"):
        match = re.fullmatch(r"module:([A-Za-z0-9][A-Za-z0-9._-]*)(?:#(.+))?", target)
        if match is None:
            _fail(f"{source_where}: malformed module link {target!r}")
        module_id, fragment = match.groups()
        if module_id not in module_paths:
            _fail(f"{source_where}: module link has unknown destination {module_id!r}")
        if fragment and fragment not in anchors_by_path[module_paths[module_id]]:
            _fail(f"{source_where}: module link has unknown anchor {fragment!r} in {module_id}")
        return

    parsed = urlsplit(target)
    if parsed.scheme:
        if parsed.scheme not in ("http", "https", "mailto"):
            _fail(f"{source_where}: unsupported link scheme {parsed.scheme!r}")
        return
    if not parsed.path:
        if parsed.fragment and parsed.fragment not in anchors_by_path[source_path]:
            _fail(f"{source_where}: unknown local anchor {parsed.fragment!r}")
        return

    candidate = source_path.parent.joinpath(*PurePosixPath(parsed.path).parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        _fail(f"{source_where}: broken internal link {target!r}")
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        _fail(f"{source_where}: internal link escapes the repository: {target!r}")
    if not resolved.is_file():
        _fail(f"{source_where}: internal link is not a file: {target!r}")
    if parsed.fragment:
        if resolved not in anchors_by_path:
            if resolved.suffix.lower() != ".md":
                _fail(f"{source_where}: anchor targets a non-Markdown file: {target!r}")
            linked_text = _read_utf8(resolved, f"linked file {resolved.relative_to(repo_root).as_posix()}")
            anchors_by_path[resolved] = {
                heading["anchor"] for heading in _headings(linked_text, str(resolved))
            }
        if parsed.fragment not in anchors_by_path[resolved]:
            _fail(f"{source_where}: unknown anchor in internal link {target!r}")


def _reject_unsupported_applied_links(markdown: str, where: str) -> list[re.Match[str]]:
    visible = HTML_COMMENT_RE.sub(" ", _without_fenced_code(markdown))
    if REFERENCE_DEFINITION_RE.search(visible) or REFERENCE_USE_RE.search(visible):
        _fail(f"{where}: reference-style links are unsupported; use inline links")
    if AUTOLINK_RE.search(visible):
        _fail(f"{where}: autolinks are unsupported; use an inline link with visible text")
    if HTML_LINK_ATTRIBUTE_RE.search(visible):
        _fail(f"{where}: HTML href/src links are unsupported; use Markdown inline links")
    matches = list(LINK_RE.finditer(visible))
    for match in matches:
        target = match.group(1) or match.group(2)
        if "(" in target or ")" in target:
            _fail(f"{where}: parentheses in link destinations are unsupported; percent-encode them")
    for marker in re.finditer(r"\]\(", visible):
        if not any(match.start() <= marker.start() < match.end() for match in matches):
            _fail(f"{where}: malformed or unsupported inline link")
    return matches


def _validate_links(
    bodies_by_path: dict[Path, str],
    module_paths: dict[str, Path],
    anchors_by_path: dict[Path, set[str]],
    repo_root: Path,
    applied_paths: set[Path],
) -> None:
    for source_path, markdown in bodies_by_path.items():
        where = source_path.relative_to(repo_root).as_posix()
        if source_path in applied_paths:
            matches = _reject_unsupported_applied_links(markdown, where)
        else:
            matches = list(LINK_RE.finditer(_without_fenced_code(markdown)))
        for match in matches:
            target = match.group(1) or match.group(2)
            _validate_link_target(target, source_path, where, module_paths, anchors_by_path, repo_root)


def _resolve_spec_root(repo_root: Path, spec_root: Path | None) -> Path:
    if spec_root is not None:
        candidate = spec_root
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        resolved = candidate.resolve(strict=True)
    else:
        authority_path = repo_root / "docs" / "implementation" / "s0" / "authority.json"
        authority = _require_object(_load_json(authority_path, "S0 authority"), "S0 authority")
        relative = _require_string(authority.get("spec_root"), "S0 authority.spec_root")
        resolved = _safe_relative_file(
            repo_root, f"{relative.rstrip('/')}/00-DECISIONS.md", "S0 authority.spec_root probe", suffix=".md"
        ).parent
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        _fail("spec root must be inside the repository")
    if not resolved.is_dir():
        _fail("spec root must be a directory")
    return resolved


def _source_file_records(paths: Iterable[Path], repo_root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(repo_root).as_posix()
        except ValueError:
            _fail(f"compiled source lies outside repository: {resolved}")
        if relative in seen:
            continue
        seen.add(relative)
        records.append({"path": relative, "sha256": _sha256_file(resolved)})
    return sorted(records, key=lambda record: record["path"])


def _validate_output_target(repo_root: Path, output_path: str | Path) -> Path:
    candidate = Path(output_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    lexical_candidate = Path(os.path.abspath(os.fspath(candidate)))
    expected = repo_root / "dist" / "applied-content.json"
    if lexical_candidate != expected:
        _fail("output path must be exactly dist/applied-content.json")
    current = repo_root
    for part in ("dist", "applied-content.json"):
        current = current / part
        is_junction = getattr(current, "is_junction", lambda: False)
        if current.is_symlink() or is_junction():
            _fail(f"output path may not contain a symlink or junction: {current}")
    return expected


def compile_edition(
    repo_root: str | Path,
    source_root: str | Path,
    output_path: str | Path | None = None,
    *,
    spec_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and compile one applied edition.

    When ``output_path`` is omitted, no filesystem mutation is performed. A
    requested output is atomically replaced only after the complete compile
    has succeeded.
    """

    repo = Path(repo_root).resolve(strict=True)
    publish_destination = (
        _validate_output_target(repo, output_path) if output_path is not None else None
    )
    source = Path(source_root).resolve(strict=True)
    try:
        source.relative_to(repo)
    except ValueError:
        _fail("source root must be inside the repository")
    active_spec = _resolve_spec_root(repo, Path(spec_root) if spec_root is not None else None)

    manifest_path = source / "manifest.json"
    answers_path = source / "answers.json"
    rubrics_path = source / "rubrics.json"
    manifest = _require_fields(
        _load_json(manifest_path, "applied manifest"),
        ("format", "edition_id", "foundation", "module_order", "applied_modules"),
        ("format", "edition_id", "foundation", "module_order", "applied_modules"),
        "applied manifest",
    )
    if manifest["format"] != SOURCE_FORMAT:
        _fail(f"applied manifest.format: expected {SOURCE_FORMAT!r}")
    if manifest["edition_id"] != "intermediate-v1":
        _fail("applied manifest.edition_id: expected 'intermediate-v1'")
    module_order = _require_list(manifest["module_order"], "applied manifest.module_order")
    if module_order != list(MODULE_ORDER):
        _fail("applied manifest.module_order: expected P00, 00 through 20, E01, E02")

    foundation_identity = _require_fields(
        manifest["foundation"],
        ("edition_id", "content_sha256"),
        ("edition_id", "content_sha256"),
        "applied manifest.foundation",
    )
    foundation_edition = _require_string(
        foundation_identity["edition_id"], "applied manifest.foundation.edition_id"
    )
    foundation_digest = _require_string(
        foundation_identity["content_sha256"], "applied manifest.foundation.content_sha256"
    )
    if not re.fullmatch(r"[0-9a-f]{64}", foundation_digest):
        _fail("applied manifest.foundation.content_sha256: expected lowercase SHA-256")

    foundation_index, foundation_curriculum_path = _validate_foundation_curriculum(repo / "course")
    foundation_curriculum = _load_json(foundation_curriculum_path, "foundation curriculum")
    if foundation_curriculum.get("edition") != foundation_edition:
        _fail("applied manifest.foundation.edition_id does not match course/curriculum.json")
    foundation_content_path = repo / "dist" / "content.json"
    if _sha256_file(foundation_content_path) != foundation_digest:
        _fail("applied manifest.foundation.content_sha256 does not match dist/content.json")

    overlay_path = active_spec / "fixtures" / "curriculum" / "curriculum-overlay.json"
    oracles_path = active_spec / "fixtures" / "curriculum" / "exercise-oracles.json"
    capstone_rubric_path = active_spec / "fixtures" / "curriculum" / "capstone-rubric.json"
    source_baseline_path = active_spec / "evidence" / "source-baseline.json"
    overlay = _require_object(_load_json(overlay_path, "curriculum overlay"), "curriculum overlay")
    if overlay.get("format") != OVERLAY_FORMAT:
        _fail(f"curriculum overlay.format: expected {OVERLAY_FORMAT!r}")
    if overlay.get("edition_id") != manifest["edition_id"]:
        _fail("curriculum overlay.edition_id does not match applied manifest")
    baseline_commit = _require_string(
        overlay.get("foundation_baseline_commit"), "curriculum overlay.foundation_baseline_commit"
    )
    if not re.fullmatch(r"[0-9a-f]{40}", baseline_commit):
        _fail("curriculum overlay.foundation_baseline_commit: expected lowercase commit SHA")
    source_baseline = _require_object(
        _load_json(source_baseline_path, "source baseline"), "source baseline"
    )
    if source_baseline.get("source_commit") != baseline_commit:
        _fail("curriculum overlay.foundation_baseline_commit does not match source baseline")
    baseline_hashes = _require_object(
        source_baseline.get("files_sha256"), "source baseline.files_sha256"
    )
    protected_foundation_paths = [
        foundation_curriculum_path,
        foundation_content_path,
        *[module["_source_path"] for module in foundation_index.values()],
    ]
    for protected_path in protected_foundation_paths:
        relative = protected_path.relative_to(repo).as_posix()
        if baseline_hashes.get(relative) != _sha256_file(protected_path):
            _fail(f"source baseline digest mismatch for {relative}")
    if overlay.get("lesson_sections") != list(LESSON_SECTIONS):
        _fail("curriculum overlay.lesson_sections does not match the fixed lesson grammar")
    if overlay.get("completion_required") != [str(value) for value in range(13, 21)]:
        _fail("curriculum overlay.completion_required must be exactly 13 through 20")
    overlay_modules = _require_list(overlay.get("modules"), "curriculum overlay.modules")
    overlay_index = _unique_index(overlay_modules, "id", "curriculum overlay.modules")
    if tuple(overlay_index) != APPLIED_IDS:
        _fail("curriculum overlay.modules: IDs/order must be P00, 13 through 20, E01, E02")
    compiled_foundation = {
        **foundation_identity,
        "baseline_commit": baseline_commit,
    }

    applied_modules = _require_list(manifest["applied_modules"], "applied manifest.applied_modules")
    applied_index = _unique_index(applied_modules, "id", "applied manifest.applied_modules")
    if tuple(applied_index) != APPLIED_IDS:
        _fail("applied manifest.applied_modules: IDs/order must be P00, 13 through 20, E01, E02")

    answers_doc = _require_fields(
        _load_json(answers_path, "answers"),
        ("format", "answers"),
        ("format", "answers"),
        "answers",
    )
    if answers_doc["format"] != ANSWERS_FORMAT:
        _fail(f"answers.format: expected {ANSWERS_FORMAT!r}")
    answer_items = _require_list(answers_doc["answers"], "answers.answers")
    answers = _unique_index(answer_items, "id", "answers.answers")
    for answer_id, answer in answers.items():
        _require_fields(
            answer,
            ("id", "module_id", "exercise_id", "oracle_id", "content"),
            ("id", "module_id", "exercise_id", "oracle_id", "content"),
            f"answer {answer_id}",
        )
        _require_string(answer["module_id"], f"answer {answer_id}.module_id")
        _require_id(answer["exercise_id"], f"answer {answer_id}.exercise_id")
        if answer["oracle_id"] is not None:
            _require_id(answer["oracle_id"], f"answer {answer_id}.oracle_id")
        _require_string(answer["content"], f"answer {answer_id}.content")

    rubrics_doc = _require_fields(
        _load_json(rubrics_path, "rubrics"),
        ("format", "rubrics"),
        ("format", "rubrics"),
        "rubrics",
    )
    if rubrics_doc["format"] != RUBRICS_FORMAT:
        _fail(f"rubrics.format: expected {RUBRICS_FORMAT!r}")
    rubric_items = _require_list(rubrics_doc["rubrics"], "rubrics.rubrics")
    rubrics = _unique_index(rubric_items, "id", "rubrics.rubrics")
    for rubric_id, rubric in rubrics.items():
        _require_fields(
            rubric,
            ("id", "module_id", "exercise_id", "content"),
            ("id", "module_id", "exercise_id", "content"),
            f"rubric {rubric_id}",
        )
        _require_string(rubric["module_id"], f"rubric {rubric_id}.module_id")
        _require_id(rubric["exercise_id"], f"rubric {rubric_id}.exercise_id")
        if not isinstance(rubric["content"], dict) or not rubric["content"]:
            _fail(f"rubric {rubric_id}.content: expected a nonempty object")

    oracles_doc = _require_object(_load_json(oracles_path, "exercise oracles"), "exercise oracles")
    if oracles_doc.get("format") != ORACLES_FORMAT:
        _fail(f"exercise oracles.format: expected {ORACLES_FORMAT!r}")
    oracle_items = _require_list(oracles_doc.get("oracles"), "exercise oracles.oracles")
    oracles = _unique_index(oracle_items, "id", "exercise oracles.oracles")
    capstone_rubric = _require_object(
        _load_json(capstone_rubric_path, "capstone rubric"), "capstone rubric"
    )

    registered_exercises: dict[str, dict[str, Any]] = {}
    used_answers: set[str] = set()
    used_rubrics: set[str] = set()
    applied_paths: dict[str, Path] = {}
    applied_bodies: dict[str, str] = {}
    compiled_exercises: dict[str, list[dict[str, Any]]] = {}

    for module_id, raw_module in applied_index.items():
        module = _require_fields(
            raw_module,
            ("id", "source", "exercises"),
            ("id", "source", "exercises"),
            f"applied module {module_id}",
        )
        expected_source = f"lessons/{module_id}.md"
        if module["source"] != expected_source:
            _fail(f"applied module {module_id}.source: expected {expected_source!r}")
        source_path = _safe_relative_file(
            source, module["source"], f"applied module {module_id}.source", suffix=".md"
        )
        applied_paths[module_id] = source_path
        body = _read_utf8(source_path, f"applied module {module_id}")
        applied_bodies[module_id] = body
        exercise_items = _require_list(module["exercises"], f"applied module {module_id}.exercises")
        if not exercise_items:
            _fail(f"applied module {module_id}.exercises: expected at least one exercise")
        module_exercises: list[dict[str, Any]] = []
        for position, raw_exercise in enumerate(exercise_items):
            exercise = _require_fields(
                raw_exercise,
                ("id", "answer_id", "rubric_id"),
                ("id", "answer_id", "rubric_id"),
                f"applied module {module_id}.exercises[{position}]",
            )
            exercise_id = _require_id(
                exercise["id"], f"applied module {module_id}.exercises[{position}].id"
            )
            if exercise_id in registered_exercises:
                _fail(f"duplicate exercise ID {exercise_id!r}")
            answer_id = exercise["answer_id"]
            rubric_id = exercise["rubric_id"]
            if answer_id is not None:
                answer_id = _require_id(answer_id, f"exercise {exercise_id}.answer_id")
            if rubric_id is not None:
                rubric_id = _require_id(rubric_id, f"exercise {exercise_id}.rubric_id")
            if answer_id is None and rubric_id is None:
                _fail(f"exercise {exercise_id}: answer_id and rubric_id cannot both be null")
            compiled = {"id": exercise_id, "answer": None, "rubric": None}
            if answer_id is not None:
                if answer_id not in answers:
                    _fail(f"exercise {exercise_id}: missing answer {answer_id!r}")
                answer = answers[answer_id]
                if answer["module_id"] != module_id or answer["exercise_id"] != exercise_id:
                    _fail(f"exercise {exercise_id}: answer {answer_id!r} has a mismatched module/exercise join")
                if answer_id in used_answers:
                    _fail(f"answer {answer_id!r} is bound more than once")
                used_answers.add(answer_id)
                compiled["answer"] = answer
            if rubric_id is not None:
                if rubric_id not in rubrics:
                    _fail(f"exercise {exercise_id}: missing rubric {rubric_id!r}")
                rubric = rubrics[rubric_id]
                if rubric["module_id"] != module_id or rubric["exercise_id"] != exercise_id:
                    _fail(f"exercise {exercise_id}: rubric {rubric_id!r} has a mismatched module/exercise join")
                if rubric_id in used_rubrics:
                    _fail(f"rubric {rubric_id!r} is bound more than once")
                used_rubrics.add(rubric_id)
                compiled["rubric"] = rubric
            registered_exercises[exercise_id] = {"module_id": module_id, "compiled": compiled}
            module_exercises.append(compiled)
        compiled_exercises[module_id] = module_exercises
        _validate_applied_markdown(
            module_id,
            _require_string(overlay_index[module_id].get("title"), f"overlay module {module_id}.title"),
            body,
            [exercise["id"] for exercise in module_exercises],
            f"applied module {module_id}",
        )

    discovered_lessons = {
        path.resolve()
        for path in (source / "lessons").rglob("*.md")
        if path.is_file()
    }
    declared_lessons = set(applied_paths.values())
    if discovered_lessons != declared_lessons:
        extra = sorted(path.relative_to(source).as_posix() for path in discovered_lessons - declared_lessons)
        missing = sorted(path.relative_to(source).as_posix() for path in declared_lessons - discovered_lessons)
        details: list[str] = []
        if extra:
            details.append("unknown source lessons " + ", ".join(extra))
        if missing:
            details.append("missing source lessons " + ", ".join(missing))
        _fail("applied lesson/source mismatch: " + "; ".join(details))
    if used_answers != set(answers):
        _fail("unbound answers: " + ", ".join(sorted(set(answers) - used_answers)))
    if used_rubrics != set(rubrics):
        _fail("unbound rubrics: " + ", ".join(sorted(set(rubrics) - used_rubrics)))

    for oracle_id, oracle in oracles.items():
        module_id = _require_string(oracle.get("module_id"), f"oracle {oracle_id}.module_id")
        registered = registered_exercises.get(oracle_id)
        if registered is None:
            _fail(f"oracle {oracle_id}: no compiled exercise has this ID")
        if registered["module_id"] != module_id:
            _fail(f"oracle {oracle_id}: compiled exercise belongs to the wrong module")
        answer = registered["compiled"]["answer"]
        if answer is None or answer["oracle_id"] != oracle_id:
            _fail(f"oracle {oracle_id}: exercise answer must bind this oracle exactly")
        registered["compiled"]["oracle"] = oracle
    known_oracle_ids = set(oracles)
    for answer_id, answer in answers.items():
        oracle_id = answer["oracle_id"]
        if oracle_id is not None and oracle_id not in known_oracle_ids:
            _fail(f"answer {answer_id}: unknown oracle {oracle_id!r}")

    module_20_rubrics = [
        exercise["rubric"]["content"]
        for exercise in compiled_exercises["20"]
        if exercise["rubric"] is not None
    ]
    if capstone_rubric not in module_20_rubrics:
        _fail("module 20: one bound rubric must exactly match the authoritative capstone rubric")

    foundation_bodies = {
        module_id: _read_utf8(module["_source_path"], f"foundation module {module_id}")
        for module_id, module in foundation_index.items()
    }
    cards = _validate_cards(overlay.get("cards"), foundation_bodies, set(MODULE_ORDER))

    module_paths = {
        **{module_id: module["_source_path"] for module_id, module in foundation_index.items()},
        **applied_paths,
    }
    bodies_by_path = {
        **{foundation_index[module_id]["_source_path"]: body for module_id, body in foundation_bodies.items()},
        **{applied_paths[module_id]: body for module_id, body in applied_bodies.items()},
    }
    anchors_by_path = {
        path: {heading["anchor"] for heading in _headings(body, path.as_posix())}
        for path, body in bodies_by_path.items()
    }
    _validate_links(bodies_by_path, module_paths, anchors_by_path, repo, set(applied_paths.values()))

    teaching_fixtures: list[Path] = []
    fixtures_dir = source / "fixtures"
    fixtures_dir_is_junction = getattr(fixtures_dir, "is_junction", lambda: False)
    if fixtures_dir.is_symlink() or fixtures_dir_is_junction():
        _fail("teaching fixture root may not be a symlink or junction")
    if fixtures_dir.exists():
        fixtures_root = fixtures_dir.resolve(strict=True)
        for path in fixtures_dir.rglob("*"):
            current = fixtures_dir
            for part in path.relative_to(fixtures_dir).parts:
                current = current / part
                current_is_junction = getattr(current, "is_junction", lambda: False)
                if current.is_symlink() or current_is_junction():
                    _fail(
                        "teaching fixture path may not contain a symlink or junction: "
                        + path.relative_to(source).as_posix()
                    )
            if path.is_file():
                resolved = path.resolve(strict=True)
                try:
                    resolved.relative_to(fixtures_root)
                except ValueError:
                    _fail(
                        "teaching fixture resolves outside the fixture root: "
                        + path.relative_to(source).as_posix()
                    )
                teaching_fixtures.append(resolved)

    input_paths = [
        manifest_path,
        answers_path,
        rubrics_path,
        foundation_curriculum_path,
        foundation_content_path,
        overlay_path,
        oracles_path,
        capstone_rubric_path,
        source_baseline_path,
        *[module["_source_path"] for module in foundation_index.values()],
        *applied_paths.values(),
        *teaching_fixtures,
    ]
    source_files = _source_file_records(input_paths, repo)
    source_digest = _sha256_bytes(
        _canonical_bytes(
            {
                "edition_id": manifest["edition_id"],
                "foundation": compiled_foundation,
                "module_order": list(MODULE_ORDER),
                "source_files": source_files,
            }
        )
    )

    compiled_modules: list[dict[str, Any]] = []
    for module_id in MODULE_ORDER:
        if module_id in foundation_index:
            source_module = foundation_index[module_id]
            metadata = {key: value for key, value in source_module.items() if key != "_source_path"}
            source_path = source_module["_source_path"]
            compiled_modules.append(
                {
                    **metadata,
                    "source_kind": "foundation",
                    "source_path": source_path.relative_to(repo).as_posix(),
                    "source_sha256": _sha256_file(source_path),
                    "body": foundation_bodies[module_id],
                    "exercises": [],
                }
            )
        else:
            metadata = dict(overlay_index[module_id])
            source_path = applied_paths[module_id]
            compiled_modules.append(
                {
                    **metadata,
                    "source_kind": "applied",
                    "source_path": source_path.relative_to(repo).as_posix(),
                    "source_sha256": _sha256_file(source_path),
                    "body": applied_bodies[module_id],
                    "exercises": compiled_exercises[module_id],
                }
            )

    compiled_output: dict[str, Any] = {
        "format": OUTPUT_FORMAT,
        "edition_id": manifest["edition_id"],
        "foundation": compiled_foundation,
        "source_digest": source_digest,
        "source_files": source_files,
        "module_order": list(MODULE_ORDER),
        "completion_required": overlay.get("completion_required"),
        "lesson_sections": list(LESSON_SECTIONS),
        "modules": compiled_modules,
        "cards": cards,
        "teaching_fixtures": [
            {
                "path": path.relative_to(repo).as_posix(),
                "sha256": _sha256_file(path),
            }
            for path in sorted(
                teaching_fixtures,
                key=lambda fixture_path: fixture_path.relative_to(repo).as_posix(),
            )
        ],
    }

    if publish_destination is not None:
        destination = _validate_output_target(repo, publish_destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = _canonical_bytes(compiled_output)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
            ) as handle:
                temporary_name = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink()
                except FileNotFoundError:
                    pass
    return compiled_output


def _parse_args(argv: list[str]) -> argparse.Namespace:
    repo_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_default)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--spec-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate and require the existing output to equal deterministic compiler bytes",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    repo = args.repo_root.resolve(strict=True)
    source = args.source_root or repo / "curriculum-applied" / "2026-09-28"
    output = args.output or repo / "dist" / "applied-content.json"
    try:
        if args.check:
            destination = _validate_output_target(repo, output)
            compiled = compile_edition(repo, source, spec_root=args.spec_root)
            payload = _canonical_bytes(compiled)
            try:
                actual = destination.read_bytes()
            except OSError as exc:
                _fail(f"compiled output is unavailable: {exc}")
            if actual != payload:
                _fail("compiled output is stale; rerun without --check")
        else:
            compiled = compile_edition(repo, source, output, spec_root=args.spec_root)
    except CompilationError as exc:
        print(f"compile_applied: ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Compiled {len(MODULE_ORDER)} modules; source digest {compiled['source_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
