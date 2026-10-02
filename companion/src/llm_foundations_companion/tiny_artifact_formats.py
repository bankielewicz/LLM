"""Framework-free validation for internal TinyLM artifact formats.

These formats are implementation details used to make S2 effects durable. They
do not amend or replace the sealed public contracts.
"""

from __future__ import annotations

import math
import re
import uuid
from collections.abc import Mapping
from typing import Any


TINY_PAIRED_EVALUATION_FORMAT = "llm-foundations-tiny-paired-evaluation-v1"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _uuid(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"{field} must be a canonical UUID") from exc
    if str(parsed) != value:
        raise ValueError(f"{field} must be a canonical UUID")
    return value


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite nonnegative number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(
            f"{field} must be a finite nonnegative number"
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be a finite nonnegative number")
    return result


def validate_tiny_paired_evaluation(value: object) -> Mapping[str, Any]:
    """Validate and return one internal two-Tiny-subject comparison artifact."""

    required = {
        "format",
        "evaluation_id",
        "job_id",
        "dataset_id",
        "dataset_manifest_sha256",
        "evaluation_profile_id",
        "records_artifact_id",
        "ordered_record_ids",
        "subjects",
        "delta_nll_per_utf8_byte",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("tiny paired evaluation has invalid fields")
    if value["format"] != TINY_PAIRED_EVALUATION_FORMAT:
        raise ValueError("tiny paired evaluation has an unsupported format")
    if value["evaluation_profile_id"] != "tiny-nll-per-byte-v1":
        raise ValueError("tiny paired evaluation has an incompatible profile")
    for field in (
        "evaluation_id",
        "job_id",
        "dataset_id",
        "records_artifact_id",
    ):
        _uuid(value[field], field)
    _digest(value["dataset_manifest_sha256"], "dataset_manifest_sha256")

    record_ids = value["ordered_record_ids"]
    if (
        not isinstance(record_ids, list)
        or not record_ids
        or len(record_ids) > 100_000
        or any(
            not isinstance(record_id, str)
            or not 1 <= len(record_id) <= 120
            for record_id in record_ids
        )
        or record_ids != sorted(record_ids)
        or len(set(record_ids)) != len(record_ids)
    ):
        raise ValueError(
            "ordered_record_ids must be unique nonempty IDs in code-point order"
        )

    subjects = value["subjects"]
    if not isinstance(subjects, list) or len(subjects) != 2:
        raise ValueError("tiny paired evaluation requires exactly two subjects")
    ratios: list[float] = []
    byte_totals: list[int] = []
    record_totals: list[int] = []
    for index, raw in enumerate(subjects):
        subject_fields = {
            "subject_index",
            "subject",
            "checkpoint_sha256",
            "tokenizer_sha256",
            "negative_log_likelihood_sum",
            "utf8_bytes_sum",
            "record_count",
            "nll_per_utf8_byte",
        }
        if not isinstance(raw, Mapping) or set(raw) != subject_fields:
            raise ValueError("tiny paired evaluation subject has invalid fields")
        if (
            isinstance(raw["subject_index"], bool)
            or not isinstance(raw["subject_index"], int)
            or raw["subject_index"] != index
        ):
            raise ValueError("tiny paired evaluation subject indexes must be [0, 1]")
        subject = raw["subject"]
        if (
            not isinstance(subject, Mapping)
            or set(subject) != {"kind", "checkpoint_id"}
            or subject.get("kind") != "tiny_checkpoint"
        ):
            raise ValueError("tiny paired evaluation subject identity is invalid")
        _uuid(subject["checkpoint_id"], f"subjects[{index}].checkpoint_id")
        _digest(raw["checkpoint_sha256"], f"subjects[{index}].checkpoint_sha256")
        _digest(raw["tokenizer_sha256"], f"subjects[{index}].tokenizer_sha256")
        nll = _finite_nonnegative(
            raw["negative_log_likelihood_sum"],
            f"subjects[{index}].negative_log_likelihood_sum",
        )
        byte_count = raw["utf8_bytes_sum"]
        record_count = raw["record_count"]
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 1
            or isinstance(record_count, bool)
            or not isinstance(record_count, int)
            or record_count != len(record_ids)
        ):
            raise ValueError("tiny paired evaluation subject denominators are invalid")
        ratio = _finite_nonnegative(
            raw["nll_per_utf8_byte"],
            f"subjects[{index}].nll_per_utf8_byte",
        )
        if ratio != nll / byte_count:
            raise ValueError("tiny paired evaluation subject ratio does not recompute")
        ratios.append(ratio)
        byte_totals.append(byte_count)
        record_totals.append(record_count)

    if len(set(byte_totals)) != 1 or len(set(record_totals)) != 1:
        raise ValueError("tiny paired subjects must cover identical source totals")
    delta = value["delta_nll_per_utf8_byte"]
    if (
        isinstance(delta, bool)
        or not isinstance(delta, (int, float))
        or not math.isfinite(float(delta))
        or float(delta) != ratios[1] - ratios[0]
    ):
        raise ValueError("tiny paired evaluation delta does not recompute")
    return value


__all__ = [
    "TINY_PAIRED_EVALUATION_FORMAT",
    "validate_tiny_paired_evaluation",
]
