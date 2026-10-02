"""Pure validation for TinyLM byte-normalized evaluation artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


_RECORD_FIELDS = frozenset(
    {
        "subject_index",
        "record_id",
        "utf8_bytes",
        "token_count",
        "negative_log_likelihood",
        "nll_per_utf8_byte",
    }
)
_METRIC_PAYLOAD_FIELDS = frozenset(
    {
        "negative_log_likelihood_sum",
        "utf8_bytes_sum",
        "record_count",
        "nll_per_utf8_byte",
        "records_artifact_id",
    }
)
_ABSOLUTE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ExpectedTinyEvaluationRecord:
    """One admitted dataset record identity needed for output validation."""

    record_id: str
    utf8_bytes: int


@dataclass(frozen=True)
class TinyEvaluationSubjectSummary:
    """Recomputed summary of one subject's ordered record rows."""

    subject_index: int
    ordered_record_ids: tuple[str, ...]
    negative_log_likelihood_sum: float
    utf8_bytes_sum: int
    record_count: int
    nll_per_utf8_byte: float


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is forbidden: {value}")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON member: {key}")
        value[key] = item
    return value


def _strict_json_object(line: bytes) -> dict[str, Any]:
    try:
        decoded = line.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("dataset JSONL is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("dataset JSONL rows must be objects")
    return value


def expected_tiny_evaluation_records(
    dataset_jsonl: bytes,
) -> tuple[ExpectedTinyEvaluationRecord, ...]:
    """Derive sorted record identities and text-byte counts from admitted bytes."""

    if not isinstance(dataset_jsonl, bytes):
        raise TypeError("dataset_jsonl must be bytes")
    lines = dataset_jsonl.splitlines()
    if not lines:
        raise ValueError("dataset JSONL must contain at least one record")
    records: list[ExpectedTinyEvaluationRecord] = []
    seen: set[str] = set()
    for line in lines:
        if not line:
            raise ValueError("dataset JSONL must not contain blank rows")
        value = _strict_json_object(line)
        record_id = value.get("record_id")
        text = value.get("text")
        if (
            not isinstance(record_id, str)
            or not record_id
            or len(record_id) > 120
            or not isinstance(text, str)
        ):
            raise ValueError(
                "dataset records require bounded record_id and text strings"
            )
        if record_id in seen:
            raise ValueError("dataset record_id values must be unique")
        try:
            utf8_bytes = len(text.encode("utf-8", errors="strict"))
        except UnicodeEncodeError as exc:
            raise ValueError("dataset text must be strict UTF-8") from exc
        if utf8_bytes <= 0:
            raise ValueError("dataset text must contain at least one UTF-8 byte")
        seen.add(record_id)
        records.append(ExpectedTinyEvaluationRecord(record_id, utf8_bytes))
    records.sort(key=lambda item: item.record_id)
    return tuple(records)


def _nonnegative_finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _same_number(left: object, right: float, label: str) -> None:
    value = _nonnegative_finite_number(left, label)
    if not math.isclose(
        value,
        right,
        rel_tol=0.0,
        abs_tol=_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError(
            f"{label} does not match recomputed evaluation records"
        )


def validate_tiny_evaluation_records(
    *,
    expected_records: Sequence[ExpectedTinyEvaluationRecord],
    record_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    subject_count: int,
    records_artifact_id: str,
) -> tuple[TinyEvaluationSubjectSummary, ...]:
    """Validate exact ordered rows and reconcile their aggregates to metrics."""

    if (
        isinstance(subject_count, bool)
        or not isinstance(subject_count, int)
        or subject_count not in {1, 2}
    ):
        raise ValueError("subject_count must be one or two")
    if not isinstance(records_artifact_id, str) or not records_artifact_id:
        raise ValueError("records_artifact_id must be a nonempty string")
    admitted = tuple(expected_records)
    if not admitted:
        raise ValueError("expected_records must not be empty")
    seen: set[str] = set()
    for item in admitted:
        if (
            not isinstance(item, ExpectedTinyEvaluationRecord)
            or not isinstance(item.record_id, str)
            or not item.record_id
            or isinstance(item.utf8_bytes, bool)
            or not isinstance(item.utf8_bytes, int)
            or item.utf8_bytes <= 0
            or item.record_id in seen
        ):
            raise ValueError("expected_records contain an invalid identity")
        seen.add(item.record_id)
    if admitted != tuple(sorted(admitted, key=lambda item: item.record_id)):
        raise ValueError(
            "expected_records must be in Unicode record_id order"
        )

    expected_row_count = len(admitted) * subject_count
    if len(record_rows) != expected_row_count:
        raise ValueError(
            "evaluation record cardinality differs from the admitted split"
        )
    if len(metric_rows) != subject_count:
        raise ValueError(
            "evaluation metric cardinality differs from its subjects"
        )

    summaries: list[TinyEvaluationSubjectSummary] = []
    offset = 0
    for subject_index in range(subject_count):
        negative_log_likelihood_sum = 0.0
        utf8_bytes_sum = 0
        ordered_ids: list[str] = []
        for expected in admitted:
            row = record_rows[offset]
            offset += 1
            if not isinstance(row, Mapping) or set(row) != _RECORD_FIELDS:
                raise ValueError("evaluation records have invalid fields")
            index = row.get("subject_index")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index != subject_index
            ):
                raise ValueError(
                    "evaluation records have invalid subject_index"
                )
            record_id = row.get("record_id")
            if not isinstance(record_id, str) or record_id != expected.record_id:
                raise ValueError(
                    "evaluation record order differs from the admitted split"
                )
            utf8_bytes = row.get("utf8_bytes")
            if (
                isinstance(utf8_bytes, bool)
                or not isinstance(utf8_bytes, int)
                or utf8_bytes != expected.utf8_bytes
            ):
                raise ValueError(
                    "evaluation record byte count differs from the admitted split"
                )
            _positive_integer(row.get("token_count"), "token_count")
            nll = _nonnegative_finite_number(
                row.get("negative_log_likelihood"),
                "negative_log_likelihood",
            )
            ratio = _nonnegative_finite_number(
                row.get("nll_per_utf8_byte"),
                "nll_per_utf8_byte",
            )
            if not math.isclose(
                ratio,
                nll / utf8_bytes,
                rel_tol=0.0,
                abs_tol=_ABSOLUTE_TOLERANCE,
            ):
                raise ValueError(
                    "record ratio does not match NLL divided by UTF-8 bytes"
                )
            ordered_ids.append(record_id)
            negative_log_likelihood_sum += nll
            utf8_bytes_sum += utf8_bytes

        aggregate = negative_log_likelihood_sum / utf8_bytes_sum
        summary = TinyEvaluationSubjectSummary(
            subject_index=subject_index,
            ordered_record_ids=tuple(ordered_ids),
            negative_log_likelihood_sum=negative_log_likelihood_sum,
            utf8_bytes_sum=utf8_bytes_sum,
            record_count=len(ordered_ids),
            nll_per_utf8_byte=aggregate,
        )
        metric = metric_rows[subject_index]
        if not isinstance(metric, Mapping):
            raise ValueError("evaluation metric rows must be objects")
        sequence = metric.get("sequence")
        step = metric.get("step")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence != subject_index
            or isinstance(step, bool)
            or not isinstance(step, int)
            or step != 0
            or metric.get("name") != "test_nll_bytes_v1"
            or metric.get("unit") != "nats_per_utf8_byte"
            or metric.get("protocol_id") != "tiny-nll-per-byte-v1"
        ):
            raise ValueError("evaluation metric identity is invalid")
        _same_number(metric.get("value"), aggregate, "metric value")
        payload = metric.get("payload")
        if (
            not isinstance(payload, Mapping)
            or set(payload) != _METRIC_PAYLOAD_FIELDS
        ):
            raise ValueError("evaluation metric payload has invalid fields")
        if payload.get("records_artifact_id") != records_artifact_id:
            raise ValueError(
                "evaluation metric does not bind the records artifact"
            )
        if (
            isinstance(payload.get("record_count"), bool)
            or payload.get("record_count") != summary.record_count
            or isinstance(payload.get("utf8_bytes_sum"), bool)
            or payload.get("utf8_bytes_sum") != summary.utf8_bytes_sum
        ):
            raise ValueError(
                "evaluation metric counts differ from evaluation records"
            )
        _same_number(
            payload.get("negative_log_likelihood_sum"),
            summary.negative_log_likelihood_sum,
            "metric negative_log_likelihood_sum",
        )
        _same_number(
            payload.get("nll_per_utf8_byte"),
            summary.nll_per_utf8_byte,
            "metric nll_per_utf8_byte",
        )
        summaries.append(summary)
    return tuple(summaries)


__all__ = [
    "ExpectedTinyEvaluationRecord",
    "TinyEvaluationSubjectSummary",
    "expected_tiny_evaluation_records",
    "validate_tiny_evaluation_records",
]
