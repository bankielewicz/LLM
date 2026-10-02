"""Focused tests for pure TinyLM evaluation-record validation."""

from __future__ import annotations

import math

import pytest

from llm_foundations_companion.tiny_evaluation_records import (
    expected_tiny_evaluation_records,
    validate_tiny_evaluation_records,
)


ARTIFACT_ID = "00000000-0000-4000-8000-000000000123"
DATASET = (
    b'{"record_id":"B","text":"eight888"}\n'
    b'{"record_id":"A","text":"abc"}\n'
)


def rows(subject_count: int = 1) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for subject_index in range(subject_count):
        values.extend(
            [
                {
                    "subject_index": subject_index,
                    "record_id": "A",
                    "utf8_bytes": 3,
                    "token_count": 2,
                    "negative_log_likelihood": 6.0,
                    "nll_per_utf8_byte": 2.0,
                },
                {
                    "subject_index": subject_index,
                    "record_id": "B",
                    "utf8_bytes": 8,
                    "token_count": 3,
                    "negative_log_likelihood": 4.0,
                    "nll_per_utf8_byte": 0.5,
                },
            ]
        )
    return values


def metrics(subject_count: int = 1) -> list[dict[str, object]]:
    return [
        {
            "sequence": subject_index,
            "step": 0,
            "name": "test_nll_bytes_v1",
            "value": 10.0 / 11.0,
            "unit": "nats_per_utf8_byte",
            "protocol_id": "tiny-nll-per-byte-v1",
            "payload": {
                "negative_log_likelihood_sum": 10.0,
                "utf8_bytes_sum": 11,
                "record_count": 2,
                "nll_per_utf8_byte": 10.0 / 11.0,
                "records_artifact_id": ARTIFACT_ID,
            },
        }
        for subject_index in range(subject_count)
    ]


def validate(
    record_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    *,
    subject_count: int,
):
    return validate_tiny_evaluation_records(
        expected_records=expected_tiny_evaluation_records(DATASET),
        record_rows=record_rows,
        metric_rows=metric_rows,
        subject_count=subject_count,
        records_artifact_id=ARTIFACT_ID,
    )


def test_single_and_paired_rows_reconcile_to_admitted_split_and_metrics() -> None:
    single = validate(rows(), metrics(), subject_count=1)
    assert single[0].ordered_record_ids == ("A", "B")
    assert single[0].utf8_bytes_sum == 11
    assert single[0].negative_log_likelihood_sum == 10.0
    assert single[0].nll_per_utf8_byte == 10.0 / 11.0
    paired = validate(rows(2), metrics(2), subject_count=2)
    assert [item.subject_index for item in paired] == [0, 1]


@pytest.mark.parametrize("invalid", [True, False, 1])
def test_subject_index_rejects_bool_and_single_subject_index_one(
    invalid: object,
) -> None:
    record_rows = rows()
    record_rows[0]["subject_index"] = invalid
    with pytest.raises(ValueError, match="subject_index"):
        validate(record_rows, metrics(), subject_count=1)


def test_rows_require_exact_fields_order_bytes_and_finite_arithmetic() -> None:
    mutations = []
    extra = rows()
    extra[0]["extra"] = 1
    mutations.append(extra)
    wrong_order = rows()
    wrong_order[0]["record_id"] = "B"
    mutations.append(wrong_order)
    wrong_bytes = rows()
    wrong_bytes[0]["utf8_bytes"] = 4
    mutations.append(wrong_bytes)
    wrong_ratio = rows()
    wrong_ratio[0]["nll_per_utf8_byte"] = 1.0
    mutations.append(wrong_ratio)
    nonfinite = rows()
    nonfinite[0]["negative_log_likelihood"] = math.inf
    mutations.append(nonfinite)
    for record_rows in mutations:
        with pytest.raises(ValueError):
            validate(record_rows, metrics(), subject_count=1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("value", 1.25),
        ("record_count", 1),
        ("utf8_bytes_sum", 10),
        ("negative_log_likelihood_sum", 9.0),
        ("nll_per_utf8_byte", 1.25),
        ("records_artifact_id", "wrong"),
    ],
)
def test_metric_totals_and_artifact_binding_are_recomputed(
    field: str, value: object
) -> None:
    metric_rows = metrics()
    if field == "value":
        metric_rows[0][field] = value
    else:
        payload = metric_rows[0]["payload"]
        assert isinstance(payload, dict)
        payload[field] = value
    with pytest.raises(ValueError):
        validate(rows(), metric_rows, subject_count=1)


def test_dataset_identity_extraction_is_strict_sorted_and_unique() -> None:
    expected = expected_tiny_evaluation_records(DATASET)
    assert [(item.record_id, item.utf8_bytes) for item in expected] == [
        ("A", 3),
        ("B", 8),
    ]
    invalid = (
        b'{"record_id":"A","record_id":"B","text":"x"}\n',
        b'{"record_id":"A","text":""}\n',
        b'{"record_id":"A","text":"x"}\n'
        b'{"record_id":"A","text":"y"}\n',
        b'{"record_id":"A","text":"\\ud800"}\n',
    )
    for payload in invalid:
        with pytest.raises(ValueError):
            expected_tiny_evaluation_records(payload)
