"""Complete byte-normalized TinyLM evaluation."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch


EOS_ID = 256


class EvaluationCancelled(RuntimeError):
    """Cancellation observed at the required post-record polling boundary."""


@dataclass(frozen=True)
class EvaluationSummary:
    records: tuple[dict[str, int | float | str], ...]
    total_negative_log_likelihood: float
    total_utf8_bytes: int
    nll_per_utf8_byte: float


def _record_fields(record: object) -> tuple[str, str]:
    if isinstance(record, Mapping):
        record_id, text = record.get("record_id"), record.get("text")
    else:
        record_id = getattr(record, "record_id", None)
        text = getattr(record, "text", None)
    if not isinstance(record_id, str) or not isinstance(text, str):
        raise ValueError("evaluation records require string record_id and text fields")
    return record_id, text


def _model_logits(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    output = model(input_ids)
    logits = output[0] if isinstance(output, tuple) else output
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("TinyLM forward must return rank-three logits")
    return logits[0, -1, :]


def aggregate_evaluation_rows(
    records: Iterable[Mapping[str, int | float | str]],
) -> tuple[float, int, float]:
    """Aggregate rows in their supplied order without averaging record ratios."""

    total_nll = 0.0
    total_bytes = 0
    count = 0
    for record in records:
        nll = record.get("negative_log_likelihood")
        byte_count = record.get("utf8_bytes")
        if (
            isinstance(nll, bool)
            or not isinstance(nll, (int, float))
            or not math.isfinite(float(nll))
            or float(nll) < 0.0
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
        ):
            raise ValueError("evaluation aggregate rows are invalid")
        total_nll += float(nll)
        total_bytes += byte_count
        count += 1
    if count == 0:
        raise ValueError("evaluation aggregate requires at least one record")
    return total_nll, total_bytes, total_nll / total_bytes


def evaluate_nll_per_byte(
    model: Any,
    tokenizer: Any,
    records: Iterable[object],
    *,
    context: int,
    device: str | torch.device,
    subject_index: int = 0,
    cancelled: Callable[[], bool] | None = None,
    eos_id: int = EOS_ID,
) -> EvaluationSummary:
    """Run the frozen tiny-nll-per-byte-v1 full-split protocol."""

    if isinstance(context, bool) or not isinstance(context, int) or context <= 0:
        raise ValueError("context must be a positive integer")
    if (
        isinstance(subject_index, bool)
        or not isinstance(subject_index, int)
        or subject_index < 0
    ):
        raise ValueError("subject_index must be a nonnegative integer")

    ordered = sorted((_record_fields(record) for record in records), key=lambda row: row[0])
    selected_device = torch.device(device)
    rows: list[dict[str, int | float | str]] = []
    model.eval()
    with torch.inference_mode():
        for record_id, text in ordered:
            utf8_bytes = len(text.encode("utf-8", errors="strict"))
            if utf8_bytes <= 0:
                raise ValueError("evaluation record text must contain at least one UTF-8 byte")
            targets = [int(value) for value in tokenizer.encode(text)]
            targets.append(eos_id)
            history = [eos_id]
            record_nll = 0.0
            for target in targets:
                input_ids = torch.tensor(
                    [history[-context:]], dtype=torch.long, device=selected_device
                )
                logits = _model_logits(model, input_ids).to(dtype=torch.float64)
                log_probabilities = torch.log_softmax(
                    logits, dim=-1, dtype=torch.float64
                )
                value = -float(log_probabilities[target].item())
                if not math.isfinite(value):
                    raise ValueError(
                        "evaluation produced a nonfinite negative log likelihood"
                    )
                record_nll += value
                history.append(target)
            ratio = record_nll / utf8_bytes
            rows.append(
                {
                    "subject_index": subject_index,
                    "record_id": record_id,
                    "utf8_bytes": utf8_bytes,
                    "token_count": len(targets),
                    "negative_log_likelihood": record_nll,
                    "nll_per_utf8_byte": ratio,
                }
            )
            if cancelled is not None and cancelled():
                raise EvaluationCancelled(
                    "evaluation cancelled after a record boundary"
                )

    total_nll, total_bytes, aggregate = aggregate_evaluation_rows(rows)
    return EvaluationSummary(tuple(rows), total_nll, total_bytes, aggregate)
