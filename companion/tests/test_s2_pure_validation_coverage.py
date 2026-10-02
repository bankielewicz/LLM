"""Failure and boundary coverage for the framework-free S2 validators."""

from __future__ import annotations

from copy import deepcopy
import math

import pytest

from llm_foundations_companion.tiny_artifact_formats import (
    TINY_PAIRED_EVALUATION_FORMAT,
    validate_tiny_paired_evaluation,
)
from llm_foundations_companion.tiny_evaluation_records import (
    ExpectedTinyEvaluationRecord,
    expected_tiny_evaluation_records,
    validate_tiny_evaluation_records,
)


EVALUATION_ID = "10000000-0000-4000-8000-000000000001"
JOB_ID = "10000000-0000-4000-8000-000000000002"
DATASET_ID = "10000000-0000-4000-8000-000000000003"
RECORDS_ID = "10000000-0000-4000-8000-000000000004"
CHECKPOINT_IDS = (
    "10000000-0000-4000-8000-000000000005",
    "10000000-0000-4000-8000-000000000006",
)


def _paired() -> dict[str, object]:
    return {
        "format": TINY_PAIRED_EVALUATION_FORMAT,
        "evaluation_id": EVALUATION_ID,
        "job_id": JOB_ID,
        "dataset_id": DATASET_ID,
        "dataset_manifest_sha256": "a" * 64,
        "evaluation_profile_id": "tiny-nll-per-byte-v1",
        "records_artifact_id": RECORDS_ID,
        "ordered_record_ids": ["record-a", "record-b"],
        "subjects": [
            {
                "subject_index": index,
                "subject": {
                    "kind": "tiny_checkpoint",
                    "checkpoint_id": CHECKPOINT_IDS[index],
                },
                "checkpoint_sha256": chr(ord("b") + index) * 64,
                "tokenizer_sha256": chr(ord("d") + index) * 64,
                "negative_log_likelihood_sum": 10.0 + 5.0 * index,
                "utf8_bytes_sum": 5,
                "record_count": 2,
                "nll_per_utf8_byte": 2.0 + index,
            }
            for index in range(2)
        ],
        "delta_nll_per_utf8_byte": 1.0,
    }


def _record_rows(subject_count: int = 1) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subject_index in range(subject_count):
        rows.extend(
            [
                {
                    "subject_index": subject_index,
                    "record_id": "record-a",
                    "utf8_bytes": 3,
                    "token_count": 2,
                    "negative_log_likelihood": 6.0,
                    "nll_per_utf8_byte": 2.0,
                },
                {
                    "subject_index": subject_index,
                    "record_id": "record-b",
                    "utf8_bytes": 8,
                    "token_count": 3,
                    "negative_log_likelihood": 4.0,
                    "nll_per_utf8_byte": 0.5,
                },
            ]
        )
    return rows


def _metric_rows(subject_count: int = 1) -> list[dict[str, object]]:
    return [
        {
            "sequence": index,
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
                "records_artifact_id": RECORDS_ID,
            },
        }
        for index in range(subject_count)
    ]


EXPECTED = (
    ExpectedTinyEvaluationRecord("record-a", 3),
    ExpectedTinyEvaluationRecord("record-b", 8),
)


def _validate_records(
    record_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    *,
    expected_records: object = EXPECTED,
    subject_count: int = 1,
    records_artifact_id: str = RECORDS_ID,
):
    return validate_tiny_evaluation_records(
        expected_records=expected_records,  # type: ignore[arg-type]
        record_rows=record_rows,
        metric_rows=metric_rows,
        subject_count=subject_count,
        records_artifact_id=records_artifact_id,
    )


def test_paired_validator_rejects_top_level_identity_and_record_bounds() -> None:
    mutations: list[object] = [None, {"format": TINY_PAIRED_EVALUATION_FORMAT}]
    for field, value in (
        ("format", "unknown"),
        ("evaluation_profile_id", "unknown"),
        ("evaluation_id", 1),
        ("job_id", "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"),
        ("dataset_id", "not-a-uuid"),
        ("records_artifact_id", ""),
        ("dataset_manifest_sha256", "A" * 64),
        ("ordered_record_ids", []),
        ("ordered_record_ids", ["record-b", "record-a"]),
        ("ordered_record_ids", ["record-a", "record-a"]),
        ("ordered_record_ids", [""]),
        ("ordered_record_ids", ["x" * 121]),
        ("ordered_record_ids", [1]),
    ):
        candidate = _paired()
        candidate[field] = value
        mutations.append(candidate)
    oversized = _paired()
    oversized["ordered_record_ids"] = [str(index) for index in range(100_001)]
    mutations.append(oversized)
    for candidate in mutations:
        with pytest.raises(ValueError):
            validate_tiny_paired_evaluation(candidate)


def test_paired_validator_rejects_subject_identity_type_confusion() -> None:
    mutations = []
    for subjects in ([], [_paired()["subjects"]], "subjects"):
        candidate = _paired()
        candidate["subjects"] = subjects
        mutations.append(candidate)
    for index, field, value in (
        (0, "subject_index", False),
        (1, "subject_index", True),
        (0, "subject_index", 1),
        (0, "subject", None),
        (0, "subject", {"kind": "other", "checkpoint_id": CHECKPOINT_IDS[0]}),
        (0, "subject", {"kind": "tiny_checkpoint", "checkpoint_id": "bad"}),
        (0, "checkpoint_sha256", "z" * 64),
        (1, "tokenizer_sha256", "e" * 63),
    ):
        candidate = _paired()
        subjects = candidate["subjects"]
        assert isinstance(subjects, list)
        subject = subjects[index]
        assert isinstance(subject, dict)
        subject[field] = value
        mutations.append(candidate)
    extra = _paired()
    subjects = extra["subjects"]
    assert isinstance(subjects, list) and isinstance(subjects[0], dict)
    subjects[0]["extra"] = None
    mutations.append(extra)
    for candidate in mutations:
        with pytest.raises(ValueError):
            validate_tiny_paired_evaluation(candidate)


def test_paired_validator_rejects_numeric_overflow_and_denominator_drift() -> None:
    mutations = []
    for index, field, value in (
        (0, "negative_log_likelihood_sum", True),
        (0, "negative_log_likelihood_sum", -1),
        (0, "negative_log_likelihood_sum", math.inf),
        (0, "negative_log_likelihood_sum", 10**400),
        (0, "utf8_bytes_sum", True),
        (0, "utf8_bytes_sum", 0),
        (0, "record_count", True),
        (0, "record_count", 1),
        (0, "nll_per_utf8_byte", math.nan),
        (0, "nll_per_utf8_byte", 1.0),
        (1, "utf8_bytes_sum", 6),
    ):
        candidate = _paired()
        subjects = candidate["subjects"]
        assert isinstance(subjects, list) and isinstance(subjects[index], dict)
        subjects[index][field] = value
        mutations.append(candidate)
    for delta in (True, math.inf, "1", 0.5):
        candidate = _paired()
        candidate["delta_nll_per_utf8_byte"] = delta
        mutations.append(candidate)
    for candidate in mutations:
        with pytest.raises(ValueError):
            validate_tiny_paired_evaluation(candidate)


def test_expected_records_reject_noncanonical_dataset_bytes() -> None:
    invalid: tuple[object, ...] = (
        "text",
        b"",
        b"\n",
        b"{",
        b"\xff",
        b"[]\n",
        b'{"text":"x"}\n',
        b'{"record_id":"","text":"x"}\n',
        ('{"record_id":"' + "x" * 121 + '","text":"x"}\n').encode(),
        b'{"record_id":"a","text":1}\n',
        b'{"record_id":"a","text":""}\n',
    )
    for payload in invalid:
        with pytest.raises((TypeError, ValueError)):
            expected_tiny_evaluation_records(payload)  # type: ignore[arg-type]


def test_record_validator_rejects_request_and_expected_identity_bounds() -> None:
    for subject_count in (True, 0, 3):
        with pytest.raises(ValueError):
            _validate_records(_record_rows(), _metric_rows(), subject_count=subject_count)
    with pytest.raises(ValueError):
        _validate_records(_record_rows(), _metric_rows(), records_artifact_id="")
    invalid_expected = (
        (),
        (object(),),
        (ExpectedTinyEvaluationRecord("", 1),),
        (ExpectedTinyEvaluationRecord("record-a", True),),
        (
            ExpectedTinyEvaluationRecord("record-a", 3),
            ExpectedTinyEvaluationRecord("record-a", 8),
        ),
        tuple(reversed(EXPECTED)),
    )
    for expected in invalid_expected:
        with pytest.raises(ValueError):
            _validate_records(
                _record_rows(), _metric_rows(), expected_records=expected
            )
    with pytest.raises(ValueError, match="cardinality"):
        _validate_records(_record_rows()[:-1], _metric_rows())
    with pytest.raises(ValueError, match="metric cardinality"):
        _validate_records(_record_rows(), [])


def test_record_validator_rejects_row_type_and_numeric_overflow() -> None:
    mutations: list[list[object]] = []
    for field, value in (
        ("token_count", True),
        ("token_count", 0),
        ("negative_log_likelihood", True),
        ("negative_log_likelihood", -1),
        ("negative_log_likelihood", 10**400),
        ("nll_per_utf8_byte", math.nan),
    ):
        rows = _record_rows()
        rows[0][field] = value
        mutations.append(rows)
    mutations.append([None, *_record_rows()[1:]])
    for rows in mutations:
        with pytest.raises(ValueError):
            _validate_records(rows, _metric_rows())  # type: ignore[arg-type]


def test_record_validator_rejects_metric_shape_and_identity_bounds() -> None:
    mutations: list[list[object]] = [[None]]
    for field, value in (
        ("sequence", True),
        ("sequence", 1),
        ("step", True),
        ("step", 1),
        ("name", "other"),
        ("unit", "bits"),
        ("protocol_id", "other"),
        ("value", 10**400),
        ("payload", None),
    ):
        metrics = _metric_rows()
        metrics[0][field] = value
        mutations.append(metrics)
    extra_payload = _metric_rows()
    payload = extra_payload[0]["payload"]
    assert isinstance(payload, dict)
    payload["extra"] = None
    mutations.append(extra_payload)
    for field, value in (
        ("record_count", True),
        ("utf8_bytes_sum", True),
        ("negative_log_likelihood_sum", 10**400),
        ("nll_per_utf8_byte", 10**400),
    ):
        metrics = _metric_rows()
        payload = metrics[0]["payload"]
        assert isinstance(payload, dict)
        payload[field] = value
        mutations.append(metrics)
    for metrics in mutations:
        with pytest.raises(ValueError):
            _validate_records(_record_rows(), metrics)  # type: ignore[arg-type]
