"""Integrity-checked, wheel-local copies of the sealed runtime contracts."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from functools import lru_cache
from importlib import resources
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping


_MANIFEST_NAME = "manifest.json"


class ContractDataError(RuntimeError):
    """The installed contract bundle is missing, malformed, or corrupt."""


def _resource_bytes(name: str) -> bytes:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ContractDataError(f"invalid bundled contract name: {name!r}")
    target = resources.files(__package__)
    for part in path.parts:
        target = target.joinpath(part)
    try:
        return target.read_bytes()
    except (FileNotFoundError, IsADirectoryError, OSError) as exc:
        raise ContractDataError(f"bundled contract is unavailable: {name}") from exc


def _plain_json(raw: bytes, name: str) -> Any:
    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractDataError(f"bundled contract is invalid JSON: {name}") from exc


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> Any:
    raise ValueError(f"non-finite number: {token}")


@lru_cache(maxsize=1)
def _manifest_rows() -> Mapping[str, Mapping[str, Any]]:
    document = _plain_json(_resource_bytes(_MANIFEST_NAME), _MANIFEST_NAME)
    if not isinstance(document, dict):
        raise ContractDataError("bundled contract manifest must be an object")
    if document.get("format") != "llm-foundations-runtime-contracts-v1":
        raise ContractDataError("unsupported bundled contract manifest format")
    rows = document.get("files")
    if not isinstance(rows, list) or not rows:
        raise ContractDataError("bundled contract manifest has no files")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ContractDataError("bundled contract manifest row must be an object")
        name = row.get("name")
        source = row.get("source")
        size = row.get("size_bytes")
        digest = row.get("sha256")
        if (
            not isinstance(name, str)
            or not isinstance(source, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or name in indexed
        ):
            raise ContractDataError("invalid bundled contract manifest row")
        indexed[name] = MappingProxyType(dict(row))
    return MappingProxyType(indexed)


@lru_cache(maxsize=None)
def _load_json_cached(name: str) -> Any:
    row = _manifest_rows().get(name)
    if row is None:
        raise ContractDataError(f"contract is not declared in the bundle: {name}")
    raw = _resource_bytes(name)
    if len(raw) != row["size_bytes"]:
        raise ContractDataError(f"bundled contract size mismatch: {name}")
    if hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise ContractDataError(f"bundled contract digest mismatch: {name}")
    return _plain_json(raw, name)


def load_json(name: str) -> Any:
    """Return a defensive copy of one declared, integrity-checked document."""

    return deepcopy(_load_json_cached(name))


def declared_files() -> tuple[str, ...]:
    """Return the deterministic bundled document inventory."""

    return tuple(sorted(_manifest_rows()))
