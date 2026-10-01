"""Resolve one concrete wheel from each sealed multi-hash profile lock.

This audit/regeneration utility reads already-pinned lock files and public
package indexes, then writes deterministic manifests. It never downloads wheel
bodies. Existing manifests remain release inputs until deliberately regenerated
and reviewed.
"""

from __future__ import annotations

import hashlib
import html.parser
import json
import functools
import re
import sys
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path

from packaging.tags import Tag, compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


LOCK_DIR = Path(__file__).resolve().parent
PYTHON_VERSION = (3, 12)
PYPI_INDEX = "https://pypi.org/simple"
PROFILE_PLATFORMS = {
    "legacy-test": tuple(
        [f"manylinux_2_{minor}_x86_64" for minor in range(28, 4, -1)]
        + ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"]
    ),
    "win-cpu": ("win_amd64",),
    "win-cuda": ("win_amd64",),
    "wsl-cpu": tuple(
        [f"manylinux_2_{minor}_x86_64" for minor in range(28, 4, -1)]
        + ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"]
    ),
    "wsl-cuda": tuple(
        [f"manylinux_2_{minor}_x86_64" for minor in range(28, 4, -1)]
        + ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"]
    ),
}
PACKAGE_START = re.compile(r"(?m)^([A-Za-z0-9_.-]+)==([^ \\\n]+)")
PACKAGE_LINE = re.compile(r"^([A-Za-z0-9_.-]+)==([^ \\]+) \\$")
HASH_LINE = re.compile(r"^    --hash=sha256:[0-9a-f]{64}(?: \\)?$")
HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})")
INDEX = re.compile(r"# from (\S+)")


@dataclass(frozen=True)
class LockedPackage:
    name: str
    version: str
    hashes: frozenset[str]
    source_index: str


@dataclass(frozen=True)
class WheelCandidate:
    filename: str
    url: str
    sha256: str
    size: int
    tags: frozenset[Tag]


class _Links(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a" and dict(attrs).get("href"):
            self.hrefs.append(dict(attrs)["href"] or "")


def validate_lock_grammar(path: Path, text: str) -> None:
    current_package: str | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        package = PACKAGE_LINE.fullmatch(line)
        if package is not None:
            try:
                Version(package.group(2))
            except ValueError as error:
                raise ValueError(
                    f"{path.name}:{line_number}: invalid exact version"
                ) from error
            current_package = canonicalize_name(package.group(1))
            continue
        if HASH_LINE.fullmatch(line) is not None:
            if current_package is None:
                raise ValueError(
                    f"{path.name}:{line_number}: hash outside package block"
                )
            continue
        raise ValueError(f"{path.name}:{line_number}: unsupported active lock line")


def parse_lock(path: Path) -> list[LockedPackage]:
    text = path.read_text(encoding="utf-8")
    validate_lock_grammar(path, text)
    matches = list(PACKAGE_START.finditer(text))
    packages: list[LockedPackage] = []
    names: set[str] = set()
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        block = text[match.start() : end]
        hashes = frozenset(HASH.findall(block))
        indexes = INDEX.findall(block)
        if not hashes:
            raise ValueError(f"{path.name}: {match.group(1)} has no SHA-256")
        if len(set(indexes)) != 1:
            raise ValueError(f"{path.name}: {match.group(1)} does not have one source index")
        name = canonicalize_name(match.group(1))
        if name in names:
            raise ValueError(f"{path.name}: duplicate package {name}")
        names.add(name)
        packages.append(
            LockedPackage(
                name=match.group(1),
                version=match.group(2),
                hashes=hashes,
                source_index=indexes[0],
            )
        )
    if not packages:
        raise ValueError(f"{path.name}: no locked packages")
    return packages


def target_tag_order(profile: str) -> dict[Tag, int]:
    platforms = PROFILE_PLATFORMS[profile]
    tags = list(
        cpython_tags(
            python_version=PYTHON_VERSION,
            abis=["cp312"],
            platforms=platforms,
        )
    )
    tags.extend(
        compatible_tags(
            python_version=PYTHON_VERSION,
            interpreter="cp312",
            platforms=platforms,
        )
    )
    return {tag: position for position, tag in enumerate(dict.fromkeys(tags))}


@functools.cache
def request_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.pypi.simple.v1+json, application/json, text/html",
            "User-Agent": "llm-foundations-s0-lock-audit/1",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


@functools.cache
def head_size(url: str) -> int:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "llm-foundations-s0-lock-audit/1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        value = response.headers.get("Content-Length")
    if value is None or not value.isdigit() or int(value) <= 0:
        raise ValueError(f"missing positive Content-Length for {url}")
    return int(value)


def candidate_from_url(url: str, sha256: str, size: int) -> WheelCandidate | None:
    clean_url, _, _ = url.partition("#")
    filename = urllib.parse.unquote(Path(urllib.parse.urlparse(clean_url).path).name)
    if not filename.endswith(".whl"):
        return None
    _, _, _, tags = parse_wheel_filename(filename)
    return WheelCandidate(
        filename=filename,
        url=clean_url,
        sha256=sha256,
        size=size,
        tags=frozenset(tags),
    )


@functools.cache
def pypi_candidates(package: LockedPackage) -> list[WheelCandidate]:
    url = (
        "https://pypi.org/pypi/"
        f"{urllib.parse.quote(canonicalize_name(package.name))}/"
        f"{urllib.parse.quote(package.version)}/json"
    )
    payload = json.loads(request_bytes(url))
    candidates: list[WheelCandidate] = []
    for artifact in payload.get("urls", []):
        if artifact.get("packagetype") != "bdist_wheel":
            continue
        candidate = candidate_from_url(
            artifact["url"],
            artifact["digests"]["sha256"],
            int(artifact["size"]),
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


@functools.cache
def pypi_artifacts_by_filename(package: LockedPackage) -> dict[str, tuple[str, int]]:
    url = (
        "https://pypi.org/pypi/"
        f"{urllib.parse.quote(canonicalize_name(package.name))}/"
        f"{urllib.parse.quote(package.version)}/json"
    )
    try:
        payload = json.loads(request_bytes(url))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return {}
        raise
    return {
        artifact["filename"]: (artifact["digests"]["sha256"], int(artifact["size"]))
        for artifact in payload.get("urls", [])
        if artifact.get("packagetype") == "bdist_wheel"
    }


@functools.cache
def simple_index_candidates(package: LockedPackage) -> list[WheelCandidate]:
    page_url = (
        package.source_index.rstrip("/")
        + "/"
        + urllib.parse.quote(canonicalize_name(package.name))
        + "/"
    )
    parser = _Links()
    parser.feed(request_bytes(page_url).decode("utf-8"))
    candidates: list[WheelCandidate] = []
    expected_version = Version(package.version)
    for href in parser.hrefs:
        url = urllib.parse.urljoin(page_url, href)
        clean_url, _, fragment = url.partition("#")
        filename = urllib.parse.unquote(Path(urllib.parse.urlparse(clean_url).path).name)
        if not filename.endswith(".whl"):
            continue
        _, version, _, _ = parse_wheel_filename(filename)
        if version != expected_version:
            continue
        digest_values = urllib.parse.parse_qs(fragment).get("sha256", [])
        if len(digest_values) == 1:
            digest = digest_values[0]
        else:
            mirrored = pypi_artifacts_by_filename(package).get(filename)
            if mirrored is not None:
                digest = mirrored[0]
            elif len(package.hashes) == 1:
                digest = next(iter(package.hashes))
            else:
                continue
        candidate = candidate_from_url(clean_url, digest, head_size(clean_url))
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def select_wheel(package: LockedPackage, tag_order: dict[Tag, int]) -> WheelCandidate:
    candidates = (
        pypi_candidates(package)
        if package.source_index == PYPI_INDEX
        else simple_index_candidates(package)
    )
    compatible: list[tuple[int, str, WheelCandidate]] = []
    for candidate in candidates:
        ranks = [tag_order[tag] for tag in candidate.tags if tag in tag_order]
        if ranks and candidate.sha256 in package.hashes:
            compatible.append((min(ranks), candidate.filename, candidate))
    if not compatible:
        raise ValueError(
            f"no compatible, lock-bound wheel for {package.name}=={package.version}"
        )
    compatible.sort(key=lambda item: (item[0], item[1]))
    return compatible[0][2]


def build_manifest(profile: str) -> dict[str, object]:
    lock_path = LOCK_DIR / f"{profile}.requirements.txt"
    rows: list[dict[str, object]] = []
    tag_order = target_tag_order(profile)
    for package in parse_lock(lock_path):
        wheel = select_wheel(package, tag_order)
        rows.append(
            {
                "name": package.name,
                "version": package.version,
                "filename": wheel.filename,
                "source_index": package.source_index,
                "url": wheel.url,
                "sha256": wheel.sha256,
                "bytes": wheel.size,
            }
        )
    return {
        "format": "llm-foundations-wheel-manifest-v1",
        "profile": profile,
        "python": "3.12",
        "platform": PROFILE_PLATFORMS[profile][0],
        "lock_file": lock_path.name,
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "package_count": len(rows),
        "total_wheel_bytes": sum(int(row["bytes"]) for row in rows),
        "packages": rows,
    }


def main() -> None:
    profiles = sys.argv[1:] or list(PROFILE_PLATFORMS)
    unknown = sorted(set(profiles) - set(PROFILE_PLATFORMS))
    if unknown:
        raise SystemExit(f"unknown profile(s): {', '.join(unknown)}")
    for profile in profiles:
        manifest = build_manifest(profile)
        output = LOCK_DIR / f"{profile}.wheels.json"
        output.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(
            f"{profile}: {manifest['package_count']} wheels, "
            f"{manifest['total_wheel_bytes']} bytes"
        )


if __name__ == "__main__":
    main()
