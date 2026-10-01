"""Offline validation for the S0 dependency declarations and lock artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
import urllib.parse
from pathlib import Path

from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version


LOCKS = Path(__file__).resolve().parent
COMPANION = LOCKS.parent
PROFILES = {
    "win-cpu": {
        "count": 39,
        "torch": "2.8.0+cpu",
        "index": "https://download.pytorch.org/whl/cpu",
        "lock_sha256": "02619fe73374e48742a6fe99317067263a106bce77580f8b79c19bca6fbb8872",
        "platforms": ("win_amd64",),
    },
    "win-cuda": {
        "count": 39,
        "torch": "2.8.0+cu128",
        "index": "https://download.pytorch.org/whl/cu128",
        "lock_sha256": "0821cbccf2b612db551d565e6420553cfc7c1aa85a61f7bddb8b699a305edb38",
        "platforms": ("win_amd64",),
    },
    "wsl-cpu": {
        "count": 38,
        "torch": "2.8.0+cpu",
        "index": "https://download.pytorch.org/whl/cpu",
        "lock_sha256": "4c49171d00476f62f80ec1cbb1fdd94c2519f1ac97d37c29c27a042605977cfb",
        "platforms": tuple(
            [f"manylinux_2_{minor}_x86_64" for minor in range(28, 4, -1)]
            + [
                "manylinux2014_x86_64",
                "manylinux2010_x86_64",
                "manylinux1_x86_64",
                "linux_x86_64",
            ]
        ),
    },
    "wsl-cuda": {
        "count": 53,
        "torch": "2.8.0+cu128",
        "index": "https://download.pytorch.org/whl/cu128",
        "lock_sha256": "e5ced4cbc9bfb327ec4917f1ae8649b8986878635c115d6f4dd2e323f40e5a52",
        "platforms": tuple(
            [f"manylinux_2_{minor}_x86_64" for minor in range(28, 4, -1)]
            + [
                "manylinux2014_x86_64",
                "manylinux2010_x86_64",
                "manylinux1_x86_64",
                "linux_x86_64",
            ]
        ),
    },
}
RUNTIME_DIRECT = {
    "accelerate": "1.10.1",
    "fastapi": "0.116.1",
    "peft": "0.17.1",
    "pydantic": "2.11.7",
    "safetensors": "0.6.2",
    "torch": "2.8.0",
    "transformers": "4.57.1",
    "uvicorn": "0.35.0",
}
DEV_DIRECT = {
    "build": "1.4.0",
    "jsonschema": "4.26.0",
    "packaging": "25.0",
    "pytest": "9.0.2",
    "referencing": "0.37.0",
    "setuptools": "68.1.2",
    "wheel": "0.42.0",
}
PROFILE_REQUIREMENTS = {
    "win-cpu": "torch==2.8.0+cpu; platform_system == 'Windows' and platform_machine == 'AMD64'",
    "win-cuda": "torch==2.8.0+cu128; platform_system == 'Windows' and platform_machine == 'AMD64'",
    "wsl-cpu": "torch==2.8.0+cpu; platform_system == 'Linux' and platform_machine == 'x86_64'",
    "wsl-cuda": "torch==2.8.0+cu128; platform_system == 'Linux' and platform_machine == 'x86_64'",
}
CUDA_TORCH = {
    "win-cuda": (
        "torch-2.8.0+cu128-cp312-cp312-win_amd64.whl",
        "0ad925202387f4e7314302a1b4f8860fa824357f9b1466d7992bf276370ebcff",
    ),
    "wsl-cuda": (
        "torch-2.8.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl",
        "4354fc05bb79b208d6995a04ca1ceef6a9547b1c4334435574353d381c55087c",
    ),
}
LEGACY_TEST = {
    "count": 10,
    "torch": "2.11.0+cpu",
    "index": "https://download.pytorch.org/whl/cpu",
    "platforms": tuple(
        [f"manylinux_2_{minor}_x86_64" for minor in range(28, 4, -1)]
        + [
            "manylinux2014_x86_64",
            "manylinux2010_x86_64",
            "manylinux1_x86_64",
            "linux_x86_64",
        ]
    ),
}
LEGACY_TORCH_WHEEL = (
    "torch-2.11.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl",
    "f82e2ae20c1545bb03997d1cc3143d94e14b800038669ee1aca45808a9acc338",
)
PACKAGE_START = re.compile(r"(?m)^([A-Za-z0-9_.-]+)==([^ \\\n]+)")
PACKAGE_LINE = re.compile(r"^([A-Za-z0-9_.-]+)==([^ \\]+) \\$")
HASH_LINE = re.compile(r"^    --hash=sha256:[0-9a-f]{64}(?: \\)?$")
HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})")
INDEX = re.compile(r"# from (\S+)")


class LockValidationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LockValidationError(message)


def exact_pins(requirements: list[str]) -> dict[str, str]:
    pins: dict[str, str] = {}
    for requirement in requirements:
        require("==" in requirement, f"dependency is not exact: {requirement}")
        name, version_and_marker = requirement.split("==", 1)
        canonical_name = canonicalize_name(name)
        require(canonical_name not in pins, f"duplicate direct dependency: {canonical_name}")
        pins[canonical_name] = version_and_marker.split(";", 1)[0].strip()
    return pins


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
            except InvalidVersion as error:
                raise LockValidationError(
                    f"{path.name}:{line_number}: invalid exact version"
                ) from error
            current_package = canonicalize_name(package.group(1))
            continue
        if HASH_LINE.fullmatch(line) is not None:
            require(
                current_package is not None,
                f"{path.name}:{line_number}: hash outside package block",
            )
            continue
        raise LockValidationError(
            f"{path.name}:{line_number}: unsupported active lock line"
        )


def parse_lock(path: Path) -> dict[str, dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    validate_lock_grammar(path, text)
    matches = list(PACKAGE_START.finditer(text))
    require(bool(matches), f"{path.name}: no packages")
    rows: dict[str, dict[str, object]] = {}
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        block = text[match.start() : end]
        name = canonicalize_name(match.group(1))
        require(name not in rows, f"{path.name}: duplicate package {name}")
        hashes = set(HASH.findall(block))
        indexes = set(INDEX.findall(block))
        require(bool(hashes), f"{path.name}: {name} has no SHA-256")
        require(len(indexes) == 1, f"{path.name}: {name} does not have one source index")
        rows[name] = {
            "version": match.group(2),
            "hashes": hashes,
            "indexes": indexes,
        }
    return rows


def supported_tags(platforms: tuple[str, ...]) -> set[object]:
    tags = set(
        cpython_tags(
            python_version=(3, 12),
            abis=["cp312"],
            platforms=platforms,
        )
    )
    tags.update(
        compatible_tags(
            python_version=(3, 12),
            interpreter="cp312",
            platforms=platforms,
        )
    )
    return tags


def validate_pyproject() -> None:
    pyproject = tomllib.loads((COMPANION / "pyproject.toml").read_text(encoding="utf-8"))
    build = pyproject["build-system"]
    require(
        build == {
            "requires": ["setuptools==68.1.2", "wheel==0.42.0"],
            "build-backend": "setuptools.build_meta",
        },
        "build-system pins changed",
    )
    project = pyproject["project"]
    require(project["requires-python"] == ">=3.12,<3.13", "Python range changed")
    require(exact_pins(project["dependencies"]) == RUNTIME_DIRECT, "D10 runtime pins changed")
    optional = project["optional-dependencies"]
    for profile, requirement in PROFILE_REQUIREMENTS.items():
        require(optional.get(profile) == [requirement], f"{profile} Torch declaration changed")
    require(exact_pins(optional["dev"]) == DEV_DIRECT, "dev direct pins changed")
    require(
        not (set(RUNTIME_DIRECT) & {"jsonschema", "referencing", "pytest", "build"}),
        "test dependency leaked into runtime",
    )


def validate_profile(profile: str, expected: dict[str, object]) -> dict[str, int]:
    lock_path = LOCKS / f"{profile}.requirements.txt"
    manifest_path = LOCKS / f"{profile}.wheels.json"
    lock_rows = parse_lock(lock_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packages = manifest["packages"]
    expected_count = int(expected["count"])
    lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()

    require(manifest["format"] == "llm-foundations-wheel-manifest-v1", f"{profile}: format")
    require(manifest["profile"] == profile, f"{profile}: profile identity")
    require(manifest["python"] == "3.12", f"{profile}: Python identity")
    require(manifest["platform"] == expected["platforms"][0], f"{profile}: platform identity")
    require(manifest["lock_file"] == lock_path.name, f"{profile}: lock filename")
    if profile in PROFILES:
        require(
            lock_sha256 == expected["lock_sha256"],
            f"{profile}: sealed lock digest",
        )
    require(
        manifest["lock_sha256"] == lock_sha256,
        f"{profile}: lock digest",
    )
    require(
        manifest["package_count"] == expected_count == len(lock_rows) == len(packages),
        f"{profile}: package count",
    )
    require(
        manifest["total_wheel_bytes"] == sum(row["bytes"] for row in packages),
        f"{profile}: byte total",
    )
    require(manifest["total_wheel_bytes"] > 0, f"{profile}: positive byte total")
    names = [canonicalize_name(row["name"]) for row in packages]
    require(names == list(lock_rows), f"{profile}: wheel rows do not match lock order")
    require(len(names) == len(set(names)), f"{profile}: duplicate manifest package")
    if profile in PROFILES:
        expected_direct = {**RUNTIME_DIRECT, "torch": str(expected["torch"])}
        for name, version in expected_direct.items():
            require(
                name in lock_rows and lock_rows[name]["version"] == version,
                f"{profile}: D10 direct pin changed: {name}",
            )

    target_tags = supported_tags(expected["platforms"])
    required_fields = {
        "name",
        "version",
        "filename",
        "source_index",
        "url",
        "sha256",
        "bytes",
    }
    for row in packages:
        require(set(row) == required_fields, f"{profile}: unexpected wheel fields")
        name = canonicalize_name(row["name"])
        locked = lock_rows[name]
        distribution, version, _, wheel_tags = parse_wheel_filename(row["filename"])
        require(canonicalize_name(distribution) == name, f"{profile}: {name} wheel name")
        require(
            version == Version(row["version"]) == Version(locked["version"]),
            f"{profile}: {name} version",
        )
        require(bool(target_tags & set(wheel_tags)), f"{profile}: {name} incompatible wheel")
        require(bool(re.fullmatch(r"[0-9a-f]{64}", row["sha256"])), f"{profile}: {name} SHA")
        require(row["sha256"] in locked["hashes"], f"{profile}: {name} SHA absent from lock")
        require({row["source_index"]} == locked["indexes"], f"{profile}: {name} source index")
        require(isinstance(row["bytes"], int) and row["bytes"] > 0, f"{profile}: {name} bytes")
        parsed_url = urllib.parse.urlparse(row["url"])
        require(not parsed_url.query and not parsed_url.fragment, f"{profile}: {name} URL suffix")
        require(
            urllib.parse.unquote(Path(parsed_url.path).name) == row["filename"],
            f"{profile}: {name} URL filename",
        )
        if row["source_index"] == "https://pypi.org/simple":
            require(parsed_url.hostname == "files.pythonhosted.org", f"{profile}: {name} PyPI host")
        else:
            source_url = urllib.parse.urlparse(row["source_index"])
            require(
                parsed_url.hostname == "files.pythonhosted.org"
                or parsed_url.hostname == "pypi.nvidia.com"
                or (
                    parsed_url.hostname
                    in {source_url.hostname, "download-r2.pytorch.org"}
                    and parsed_url.path.startswith("/whl/")
                ),
                f"{profile}: {name} non-PyPI source",
            )

    torch = next(row for row in packages if canonicalize_name(row["name"]) == "torch")
    require(torch["version"] == expected["torch"], f"{profile}: Torch version")
    require(torch["source_index"] == expected["index"], f"{profile}: Torch source")
    if profile in CUDA_TORCH:
        require(
            (torch["filename"], torch["sha256"]) == CUDA_TORCH[profile],
            f"{profile}: CUDA Torch wheel differs from sealed specification",
        )
    return {
        "package_count": expected_count,
        "total_wheel_bytes": int(manifest["total_wheel_bytes"]),
    }


def validate_dev_lock() -> int:
    rows = parse_lock(LOCKS / "dev.requirements.txt")
    for name, version in DEV_DIRECT.items():
        require(name in rows and rows[name]["version"] == version, f"dev pin changed: {name}")
    require(not (set(rows) & set(RUNTIME_DIRECT)), "runtime dependency present in dev lock")
    require("torch" not in rows, "Torch present in dev lock")
    for name, row in rows.items():
        require(row["indexes"] == {"https://pypi.org/simple"}, f"dev source changed: {name}")
    return len(rows)


def validate_legacy_lock() -> dict[str, int]:
    course_requirements = [
        line.strip()
        for line in (COMPANION.parent / "course" / "requirements.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    legacy_input = [
        line.strip()
        for line in (LOCKS / "legacy-test.in").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    require(course_requirements == ["torch==2.11.0"], "legacy course Torch pin changed")
    require(legacy_input == ["torch==2.11.0+cpu"], "legacy test Torch build changed")
    result = validate_profile("legacy-test", LEGACY_TEST)
    manifest = json.loads((LOCKS / "legacy-test.wheels.json").read_text(encoding="utf-8"))
    torch = next(row for row in manifest["packages"] if row["name"] == "torch")
    require(
        (torch["filename"], torch["sha256"]) == LEGACY_TORCH_WHEEL,
        "legacy-test: Torch wheel identity changed",
    )
    return result


def validate() -> dict[str, object]:
    validate_pyproject()
    profiles = {
        profile: validate_profile(profile, expected)
        for profile, expected in PROFILES.items()
    }
    return {
        "status": "PASS",
        "profiles": profiles,
        "dev_package_count": validate_dev_lock(),
        "legacy_test": validate_legacy_lock(),
    }


def validate_wheelhouse(profile: str, directory: Path) -> dict[str, int]:
    require(profile in {*PROFILES, "legacy-test"}, f"unknown wheelhouse profile: {profile}")
    require(directory.is_dir(), f"wheelhouse is not a directory: {directory}")
    manifest = json.loads((LOCKS / f"{profile}.wheels.json").read_text(encoding="utf-8"))
    expected = {row["filename"]: row for row in manifest["packages"]}
    actual = {path.name: path for path in directory.iterdir() if path.is_file()}
    require(set(actual) == set(expected), f"{profile}: wheelhouse file set differs from manifest")
    total_bytes = 0
    for filename, row in expected.items():
        path = actual[filename]
        size = path.stat().st_size
        require(size == row["bytes"], f"{profile}: {filename} byte count")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        require(digest.hexdigest() == row["sha256"], f"{profile}: {filename} SHA-256")
        total_bytes += size
    require(total_bytes == manifest["total_wheel_bytes"], f"{profile}: wheelhouse byte total")
    return {"package_count": len(expected), "total_wheel_bytes": total_bytes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheelhouse",
        nargs=2,
        metavar=("PROFILE", "DIRECTORY"),
        help="also verify an exact wheelhouse against one committed manifest",
    )
    arguments = parser.parse_args()
    try:
        summary = validate()
        if arguments.wheelhouse:
            profile, directory = arguments.wheelhouse
            summary["wheelhouse"] = validate_wheelhouse(profile, Path(directory))
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
