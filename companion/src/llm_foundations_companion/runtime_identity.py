"""Framework-free installed dependency and build identities."""
from __future__ import annotations

import base64
import hashlib
import json
from importlib import metadata
from pathlib import Path
import re
from importlib.resources import files


def _read(name):
    return json.loads(files("llm_foundations_companion").joinpath("runtime_data", name).read_text(encoding="utf-8"))


def profile_lock(profile: str) -> str:
    value = _read("runtime-lock-manifest.json")
    if value.get("format") != "llmf-runtime-lock-manifest-v1" or set(value.get("profiles", {})) != {"win-cpu", "win-cuda", "wsl-cpu", "wsl-cuda"}:
        raise ValueError("Invalid packaged runtime lock identities")
    digest = value["profiles"][profile]["dependency_lock_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Invalid dependency lock digest")
    return digest


def _verify_installed_source(value: dict) -> None:
    """Bind the loaded module location and exact source inventory to a wheel."""
    root = Path(__file__).resolve().parent
    try:
        distribution = metadata.distribution("llm-foundations-companion")
    except metadata.PackageNotFoundError as exc:
        raise ValueError("Build provenance requires an installed companion wheel") from exc
    if (
        Path(distribution.locate_file("llm_foundations_companion")).resolve() != root
        or not distribution.read_text("WHEEL")
        or not distribution.files
    ):
        raise ValueError("Loaded companion source is not the installed wheel")
    recorded = {str(item): item for item in distribution.files}
    rows = []
    observed = set()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Installed companion source contains a link")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Installed companion source is not a regular file")
        relative = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        member = "llm_foundations_companion/" + relative
        observed.add(member)
        entry = recorded.get(member)
        raw = path.read_bytes()
        file_hash = hashlib.sha256(raw).digest()
        if (
            entry is None or entry.hash is None or entry.hash.mode != "sha256"
            or entry.hash.value != base64.urlsafe_b64encode(file_hash).rstrip(b"=").decode("ascii")
            or entry.size != len(raw)
        ):
            raise ValueError("Installed companion source differs from its wheel record")
        if relative != "runtime_data/build-provenance.json":
            rows.append({"path": relative, "sha256": file_hash.hex(), "size_bytes": len(raw)})
    expected = {name for name in recorded if name.startswith("llm_foundations_companion/")
                and "__pycache__" not in Path(name).parts and not name.endswith(".pyc")}
    if observed != expected:
        raise ValueError("Installed companion source inventory differs from its wheel")
    rows.sort(key=lambda row: row["path"])
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != value["package_source_sha256"]:
        raise ValueError("Installed companion source differs from its build identity")


def build_provenance() -> dict | None:
    try:
        value = _read("build-provenance.json")
    except FileNotFoundError:
        return None
    if set(value) != {"format", "source_revision", "source_tree", "dirty", "package_source_sha256"} or value["format"] != "llmf-build-provenance-v1":
        raise ValueError("Invalid installed build provenance")
    for field in ("source_revision", "source_tree"):
        if not isinstance(value[field], str) or re.fullmatch(r"[0-9a-f]{40}", value[field]) is None:
            raise ValueError("Invalid installed source revision")
    if type(value["dirty"]) is not bool or not isinstance(value["package_source_sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", value["package_source_sha256"]) is None:
        raise ValueError("Invalid installed package digest")
    _verify_installed_source(value)
    return value


def source_revision() -> str | None:
    value = build_provenance()
    return None if value is None else value["source_revision"]
