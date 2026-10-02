"""Framework-free TinyLM checkpoint metadata parsing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def _strict_object(raw: bytes, label: str) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains a duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains a nonfinite number")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def load_config_metadata(raw: bytes) -> tuple[dict[str, int], str, str]:
    """Validate config.json without importing the model framework."""

    if not isinstance(raw, bytes):
        raise TypeError("config.json must be bytes")
    value = _strict_object(raw, "config.json")
    required = {
        "format",
        "architecture_profile_id",
        "vocab_size",
        "context",
        "width",
        "heads",
        "layers",
    }
    if set(value) != required or value.get("format") != "tiny-v2-config-v1":
        raise ValueError("config.json has invalid fields or format")
    architecture = value["architecture_profile_id"]
    if architecture not in {
        "tiny-v2-standard-v1",
        "tiny-v2-weight-tied-v1",
    }:
        raise ValueError("config.json has an unknown architecture profile")
    limits = {
        "vocab_size": (257, 1024),
        "context": (8, 512),
        "width": (16, 512),
        "heads": (1, 16),
        "layers": (1, 12),
    }
    config: dict[str, int] = {}
    for name, (minimum, maximum) in limits.items():
        item = value[name]
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or not minimum <= item <= maximum
        ):
            raise ValueError(f"config.json field {name} is outside its bounds")
        config[name] = item
    if config["width"] % config["heads"]:
        raise ValueError("config.json width must divide evenly by heads")
    return config, architecture, hashlib.sha256(raw).hexdigest()


__all__ = ["load_config_metadata"]
