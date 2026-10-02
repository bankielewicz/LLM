"""Worker operations for TinyLM evaluation, preview, and generation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..errors import ApiError
from ..operations import OperationFailure
from ..schema import canonical_json, strict_json
from ..tiny_artifact_formats import (
    TINY_PAIRED_EVALUATION_FORMAT,
    validate_tiny_paired_evaluation,
)
from ..tiny_v2.data import parse_jsonl_documents
from ..tiny_v2.metadata import load_config_metadata
from ..tiny_v2.preview import preview_prompt
from ..tiny_v2.tokenizer import load_tokenizer_bytes


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _checkpoint_files(context: Any, prefix: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for item in context.snapshot["inputs"]:
        role = item["role"]
        if role.startswith(prefix):
            files[role[len(prefix) :]] = context.input(role).path
    return files


def _checkpoint_descriptor(value: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = value.get("checkpoint")
    return nested if isinstance(nested, Mapping) else value


def _expectations(checkpoint: Mapping[str, Any]) -> Any:
    from ..tiny_v2.checkpoint import CheckpointExpectations

    return CheckpointExpectations(
        manifest_sha256=checkpoint.get("manifest_sha256")
        or checkpoint.get("checkpoint_sha256")
        or checkpoint.get("sha256"),
        tokenizer_sha256=checkpoint.get("tokenizer_sha256"),
        config_sha256=checkpoint.get("config_sha256"),
        architecture_profile_id=checkpoint.get("architecture_profile_id"),
    )


def _load_for_inference(
    context: Any,
    *,
    prefix: str,
    checkpoint: Mapping[str, Any],
) -> Any:
    from ..tiny_v2.checkpoint import CheckpointIncompatible, load_checkpoint

    try:
        return load_checkpoint(
            _checkpoint_files(context, prefix),
            device=context.snapshot["device"],
            expected=_expectations(checkpoint),
            for_resume=False,
        )
    except CheckpointIncompatible as exc:
        raise OperationFailure(
            "CHECKPOINT_INCOMPATIBLE",
            "The verified TinyLM checkpoint is incompatible with this operation.",
        ) from exc


def _subject_identity(
    checkpoint: Mapping[str, Any],
    context: Any,
    *,
    context_limit: int,
) -> tuple[dict[str, Any], str]:
    checkpoint_sha256 = (
        checkpoint.get("manifest_sha256")
        or checkpoint.get("checkpoint_sha256")
        or checkpoint.get("sha256")
    )
    identity = {
        "backend": "tiny",
        "model_id": checkpoint.get("model_id"),
        "checkpoint_sha256": checkpoint_sha256,
        "runtime_profile": context.runtime_profile,
        "device": context.snapshot["device"],
        "context_limit": context_limit,
    }
    if not isinstance(identity["model_id"], str) or not isinstance(
        identity["checkpoint_sha256"], str
    ):
        raise ValueError("resolved checkpoint identity is incomplete")
    return identity, hashlib.sha256(canonical_json(identity)).hexdigest()


def _generation_request_from_preview(request: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(request)
    value["operation"] = "generate"
    value.pop("backend", None)
    return value


def _preview_field(
    request: Mapping[str, Any],
    context: Any,
    *,
    tokenizer: Any,
    context_limit: int,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    prompt = preview_prompt(tokenizer, request["prompt"], context=context_limit)
    _identity, subject_digest = _subject_identity(
        checkpoint, context, context_limit=context_limit
    )
    generation_request = _generation_request_from_preview(request)
    return {
        "backend": "tiny",
        "subject_identity_sha256": subject_digest,
        "tokenizer_sha256": tokenizer.fingerprint(),
        "template_sha256": None,
        "device": context.snapshot["device"],
        "original_input_token_count": prompt.original_input_token_count,
        "input_token_ids": list(prompt.input_token_ids),
        "input_token_count": prompt.input_token_count,
        "serialized_text": prompt.serialized_text,
        "dropped_message_indices": [],
        "cropped_input_tokens": prompt.cropped_input_tokens,
        "effective_context_budget": prompt.effective_context_budget,
        "canonical_generation_request_sha256": hashlib.sha256(
            canonical_json(generation_request)
        ).hexdigest(),
    }


def _jsonl(rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]) -> bytes:
    return b"".join(canonical_json(dict(row)) + b"\n" for row in rows)


def _tiny_pair(
    request: Mapping[str, Any],
    context: Any,
    resolved_subjects: list[Mapping[str, Any]],
    summaries: list[Any],
    records_artifact_id: str,
    evaluation_id: str,
) -> dict[str, Any]:
    record_ids = [str(row["record_id"]) for row in summaries[0].records]
    if [str(row["record_id"]) for row in summaries[1].records] != record_ids:
        raise ValueError("paired evaluation record order differs between subjects")
    subjects = []
    for index, summary in enumerate(summaries):
        checkpoint = _checkpoint_descriptor(resolved_subjects[index])
        subjects.append(
            {
                "subject_index": index,
                "subject": dict(request["subjects"][index]),
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "tokenizer_sha256": checkpoint["tokenizer_sha256"],
                "negative_log_likelihood_sum": (
                    summary.total_negative_log_likelihood
                ),
                "utf8_bytes_sum": summary.total_utf8_bytes,
                "record_count": len(summary.records),
                "nll_per_utf8_byte": summary.nll_per_utf8_byte,
            }
        )
    value = {
        "format": TINY_PAIRED_EVALUATION_FORMAT,
        "evaluation_id": evaluation_id,
        "job_id": context.job_id,
        "dataset_id": request["dataset_id"],
        "dataset_manifest_sha256": context.resolved[
            "dataset_manifest_sha256"
        ],
        "evaluation_profile_id": "tiny-nll-per-byte-v1",
        "records_artifact_id": records_artifact_id,
        "ordered_record_ids": record_ids,
        "subjects": subjects,
        "delta_nll_per_utf8_byte": (
            summaries[1].nll_per_utf8_byte
            - summaries[0].nll_per_utf8_byte
        ),
    }
    validate_tiny_paired_evaluation(value)
    return value


def handle_evaluate(
    request: Mapping[str, Any],
    context: Any,
) -> Mapping[str, Any]:
    """Evaluate one or two TinyLM checkpoints over the complete selected split."""

    import torch

    from ..tiny_v2.evaluation import EvaluationCancelled, evaluate_nll_per_byte

    if request["evaluation_profile_id"] != "tiny-nll-per-byte-v1":
        raise OperationFailure(
            "SUBJECT_INCOMPATIBLE",
            "This build slice supports only TinyLM byte-normalized evaluation.",
        )
    split = request["split"]
    records = parse_jsonl_documents(
        context.input(f"dataset.{split}").path.read_bytes(),
        split=split,
    )
    context.emit("phase_changed", {"phase": "evaluation"})
    resolved_subjects = list(context.resolved["subjects"])
    all_rows: list[Mapping[str, Any]] = []
    summaries: list[Any] = []

    try:
        for index, subject in enumerate(request["subjects"]):
            if subject["kind"] != "tiny_checkpoint":
                raise OperationFailure(
                    "SUBJECT_INCOMPATIBLE",
                    "This build slice evaluates only TinyLM checkpoints.",
                )
            resolved = _checkpoint_descriptor(resolved_subjects[index])
            loaded = _load_for_inference(
                context,
                prefix=f"subject.{index}.checkpoint.",
                checkpoint=resolved,
            )
            summary = evaluate_nll_per_byte(
                loaded.model,
                loaded.tokenizer,
                records,
                context=loaded.config.context,
                device=context.snapshot["device"],
                subject_index=index,
                cancelled=lambda: context.cancellation.cancelled,
            )
            summaries.append(summary)
            all_rows.extend(summary.records)
    except EvaluationCancelled:
        context.interrupt(context.cancellation.reason or "user_cancelled")
    except torch.cuda.OutOfMemoryError as exc:
        raise OperationFailure(
            "OUT_OF_MEMORY",
            "Tiny-v2 evaluation ran out of memory.",
        ) from exc

    records_artifact = context.stage_artifact(
        "evaluation_records",
        _jsonl(all_rows),
        filename="evaluation-records.jsonl",
    )
    run_id = context.output_allocations["run_id"]
    if run_id is None:
        raise ValueError("evaluate has no parent-allocated run identity")
    recorded_at = _now()
    metric_rows = []
    for sequence, summary in enumerate(summaries):
        metric_rows.append(
            {
                "run_id": run_id,
                "sequence": sequence,
                "step": 0,
                "name": "test_nll_bytes_v1",
                "value": summary.nll_per_utf8_byte,
                "unit": "nats_per_utf8_byte",
                "protocol_id": "tiny-nll-per-byte-v1",
                "recorded_at": recorded_at,
                "payload": {
                    "negative_log_likelihood_sum": (
                        summary.total_negative_log_likelihood
                    ),
                    "utf8_bytes_sum": summary.total_utf8_bytes,
                    "record_count": len(summary.records),
                    "nll_per_utf8_byte": summary.nll_per_utf8_byte,
                    "records_artifact_id": records_artifact["artifact_id"],
                },
            }
        )
    metrics_artifact = context.stage_artifact(
        "evaluation_metrics",
        _jsonl(metric_rows),
        filename="evaluation-metrics.jsonl",
    )
    artifact_ids = [
        records_artifact["artifact_id"],
        metrics_artifact["artifact_id"],
    ]
    paired_artifact_id = None
    if len(summaries) == 2:
        pair = _tiny_pair(
            request,
            context,
            resolved_subjects,
            summaries,
            records_artifact["artifact_id"],
            run_id,
        )
        paired = context.stage_artifact(
            "evaluation_paired",
            canonical_json(pair),
            filename="tiny-paired-evaluation.json",
        )
        paired_artifact_id = paired["artifact_id"]
        artifact_ids.append(paired_artifact_id)
    return {
        "operation": "evaluate",
        "evaluation_id": run_id,
        "subjects": [dict(value) for value in request["subjects"]],
        "metrics_artifact_id": metrics_artifact["artifact_id"],
        "records_artifact_id": records_artifact["artifact_id"],
        "paired_artifact_id": paired_artifact_id,
        "artifact_ids": artifact_ids,
    }


def handle_context_preview(
    request: Mapping[str, Any],
    context: Any,
) -> Mapping[str, Any]:
    """Compute the immutable TinyLM prompt preview without importing Torch."""

    if request["backend"] != "tiny":
        raise OperationFailure(
            "SUBJECT_INCOMPATIBLE",
            "This build slice supports only TinyLM context preview.",
        )
    tokenizer = load_tokenizer_bytes(
        context.input("checkpoint.tokenizer.json").path.read_bytes()
    )
    config, _architecture, config_sha256 = load_config_metadata(
        context.input("checkpoint.config.json").path.read_bytes()
    )
    checkpoint = _checkpoint_descriptor(context.resolved["checkpoint"])
    if checkpoint.get("config_sha256") != config_sha256:
        raise OperationFailure(
            "CHECKPOINT_INCOMPATIBLE",
            "The checkpoint configuration identity changed before preview.",
        )
    if checkpoint.get("tokenizer_sha256") != tokenizer.fingerprint():
        raise OperationFailure(
            "CHECKPOINT_INCOMPATIBLE",
            "The checkpoint tokenizer identity changed before preview.",
        )
    field = _preview_field(
        request,
        context,
        tokenizer=tokenizer,
        context_limit=config["context"],
        checkpoint=checkpoint,
    )
    preview_digest = hashlib.sha256(canonical_json(field)).hexdigest()
    artifact = context.stage_artifact(
        "context_preview",
        canonical_json(field),
        filename="context-preview.json",
    )
    return {
        "operation": "context_preview",
        "preview_artifact_id": artifact["artifact_id"],
        **field,
        "context_preview_digest": preview_digest,
        "artifact_ids": [artifact["artifact_id"]],
    }


def _stale(message: str) -> OperationFailure:
    return OperationFailure("CONTEXT_PREVIEW_STALE", message)


def handle_generate(
    request: Mapping[str, Any],
    context: Any,
) -> Mapping[str, Any]:
    """Reproduce a preview before importing Torch or reading learned weights."""

    checkpoint = _checkpoint_descriptor(context.resolved["checkpoint"])
    tokenizer = load_tokenizer_bytes(
        context.input("checkpoint.tokenizer.json").path.read_bytes()
    )
    config, _architecture, config_sha256 = load_config_metadata(
        context.input("checkpoint.config.json").path.read_bytes()
    )
    if checkpoint.get("config_sha256") != config_sha256:
        raise _stale("The checkpoint configuration no longer matches the preview.")
    if checkpoint.get("tokenizer_sha256") != tokenizer.fingerprint():
        raise _stale("The checkpoint tokenizer no longer matches the preview.")
    preview_request = {
        key: value
        for key, value in request.items()
        if key not in {"preview_artifact_id", "context_preview_digest"}
    }
    preview_request["operation"] = "context_preview"
    preview_request["backend"] = "tiny"
    reproduced = _preview_field(
        preview_request,
        context,
        tokenizer=tokenizer,
        context_limit=config["context"],
        checkpoint=checkpoint,
    )
    preview_input = context.input("context_preview")
    if preview_input.artifact_id != request["preview_artifact_id"]:
        raise _stale("The referenced preview artifact changed.")
    try:
        stored = strict_json(preview_input.path.read_bytes())
    except (ApiError, TypeError, ValueError, UnicodeError) as exc:
        raise _stale("The referenced preview artifact is invalid.") from exc
    if (
        not isinstance(stored, Mapping)
        or dict(stored) != reproduced
        or hashlib.sha256(canonical_json(dict(stored))).hexdigest()
        != request["context_preview_digest"]
    ):
        raise _stale("The generation request no longer matches its context preview.")

    import torch

    from ..tiny_v2.decoding import generate_tokens

    try:
        loaded = _load_for_inference(
            context,
            prefix="checkpoint.",
            checkpoint=checkpoint,
        )
        context.emit("phase_changed", {"phase": "generating"})
        started_at = _now()
        generated = generate_tokens(
            loaded.model,
            loaded.tokenizer,
            request["prompt"],
            context=loaded.config.context,
            max_new_tokens=request["max_new_tokens"],
            temperature=request["temperature"],
            top_p=request["top_p"],
            seed=request["seed"],
            device=context.snapshot["device"],
            cancelled=lambda: context.cancellation.cancelled,
        )
        ended_at = _now()
    except torch.cuda.OutOfMemoryError as exc:
        raise OperationFailure(
            "OUT_OF_MEMORY",
            "Tiny-v2 generation ran out of memory.",
        ) from exc
    run_id = context.output_allocations["run_id"]
    if run_id is None:
        raise ValueError("generate has no parent-allocated run identity")
    generation_artifact = {
        "format": "llm-foundations-generation-v1",
        "job_id": context.job_id,
        "run_id": run_id,
        "backend": "tiny",
        "subject": {"checkpoint_id": request["checkpoint_id"]},
        "subject_identity_sha256": reproduced["subject_identity_sha256"],
        "preview_artifact_id": request["preview_artifact_id"],
        "context_preview_digest": request["context_preview_digest"],
        "input": {"prompt": request["prompt"]},
        "serialized_input": generated.preview.serialized_text,
        "input_token_ids": list(generated.preview.input_token_ids),
        "input_token_count": generated.preview.input_token_count,
        "original_input_token_count": (
            generated.preview.original_input_token_count
        ),
        "dropped_message_indices": [],
        "cropped_input_tokens": generated.preview.cropped_input_tokens,
        "effective_context_budget": generated.preview.effective_context_budget,
        "decoding": {
            "max_new_tokens": request["max_new_tokens"],
            "temperature": request["temperature"],
            "top_p": request["top_p"],
            "seed": request["seed"],
        },
        "output_token_ids": list(generated.generated_token_ids),
        "generated_token_count": generated.generated_token_count,
        "text": generated.generated_text,
        "stop_reason": generated.stop_reason,
        "utf8_replacement_occurred": generated.replacement_occurred,
        "conversation": None,
        "retrieval": None,
        "started_at": started_at,
        "ended_at": ended_at,
    }
    artifact = context.stage_artifact(
        "generation",
        canonical_json(generation_artifact),
        filename="generation.json",
    )
    if generated.stop_reason == "cancelled":
        context.interrupt(context.cancellation.reason or "user_cancelled")
    return {
        "operation": "generate",
        "run_id": run_id,
        "checkpoint_id": request["checkpoint_id"],
        "generated_text": generated.generated_text,
        "generated_token_count": generated.generated_token_count,
        "stop_reason": generated.stop_reason,
        "serialized_input": generated.preview.serialized_text,
        "truncated_input_tokens": generated.preview.cropped_input_tokens,
        "artifact_ids": [artifact["artifact_id"]],
    }


__all__ = [
    "handle_context_preview",
    "handle_evaluate",
    "handle_generate",
]
