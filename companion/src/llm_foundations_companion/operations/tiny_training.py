"""Worker-only tiny_train and tiny_resume operation handlers."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import OperationFailure


def _configure_runtime(seed: int, device: str) -> Any:
    if device == "cuda":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from ..tiny_v2.determinism import configure_torch_determinism

    try:
        return configure_torch_determinism(seed, device)
    except RuntimeError as exc:
        raise OperationFailure(
            "DETERMINISTIC_KERNEL_UNAVAILABLE",
            "The selected profile cannot provide deterministic Tiny-v2 execution.",
        ) from exc


def _dataset_bindings(resolved: Mapping[str, Any]) -> tuple[Any, Any]:
    from ..tiny_v2.training import DatasetBinding

    values = tuple(
        DatasetBinding(
            split=str(item["split"]),
            dataset_id=str(item["dataset_id"]),
            sha256=str(item["sha256"]),
        )
        for item in resolved["dataset_bindings"]
    )
    by_split = {item.split: item for item in values}
    if set(by_split) != {"train", "validation"}:
        raise OperationFailure(
            "WORKER_PROTOCOL_ERROR",
            "The verified Tiny-v2 snapshot lacks train or validation identity.",
        )
    return by_split["train"], by_split["validation"]


def _read_dataset(context: Any, split: str) -> Any:
    from ..tiny_v2.data import parse_jsonl_documents

    try:
        return parse_jsonl_documents(
            context.input(f"dataset.{split}").path.read_bytes(), split=split
        )
    except (OSError, TypeError, ValueError) as exc:
        raise OperationFailure(
            "WORKER_PROTOCOL_ERROR",
            f"The verified {split} dataset could not be parsed.",
        ) from exc


def _encode_splits(
    train_documents: Any,
    validation_documents: Any,
    tokenizer: Any,
    context_size: int,
) -> tuple[Any, Any]:
    from ..tiny_v2.data import encode_documents

    try:
        return (
            encode_documents(
                train_documents, tokenizer, context_size, split="train"
            ),
            encode_documents(
                validation_documents,
                tokenizer,
                context_size,
                split="validation",
            ),
        )
    except ValueError as exc:
        if "Need at least" in str(exc):
            raise OperationFailure(
                "SPLIT_TOO_SHORT",
                "A Tiny-v2 split is not longer than the model context.",
            ) from exc
        raise OperationFailure(
            "WORKER_PROTOCOL_ERROR",
            "Verified Tiny-v2 dataset bytes violate the worker contract.",
        ) from exc


def _metrics_bytes(run_id: str, metrics: tuple[Any, ...]) -> bytes:
    from ..tiny_v2.training import Evaluation, StepMetric

    rows: list[dict[str, Any]] = []
    sequence = 0
    recorded_at = (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )

    def append(
        *,
        step: int,
        name: str,
        value: float,
        unit: str,
        protocol_id: str,
    ) -> None:
        nonlocal sequence
        rows.append(
            {
                "run_id": run_id,
                "sequence": sequence,
                "step": step,
                "name": name,
                "value": value,
                "unit": unit,
                "protocol_id": protocol_id,
                "recorded_at": recorded_at,
            }
        )
        sequence += 1

    for item in metrics:
        if isinstance(item, StepMetric):
            append(
                step=item.step,
                name="train_nll_token",
                value=item.train_nll_token,
                unit="nats_per_token",
                protocol_id="tiny-v2-training-v1",
            )
            append(
                step=item.step,
                name="gradient_l2_norm",
                value=item.gradient_l2_norm,
                unit="l2_norm",
                protocol_id="tiny-v2-training-v1",
            )
            append(
                step=item.step,
                name="elapsed_seconds",
                value=item.elapsed_seconds,
                unit="seconds",
                protocol_id="tiny-v2-training-v1",
            )
        elif isinstance(item, Evaluation):
            append(
                step=item.step,
                name="train_nll_token",
                value=item.train_nll_token,
                unit="nats_per_token",
                protocol_id="tiny-v2-fixed-eight-batches-v1",
            )
            append(
                step=item.step,
                name="validation_nll_token",
                value=item.validation_nll_token,
                unit="nats_per_token",
                protocol_id="tiny-v2-fixed-eight-batches-v1",
            )
        else:
            raise OperationFailure(
                "WORKER_PROTOCOL_ERROR", "The trainer produced an unknown metric row."
            )
    return b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _make_callbacks(
    *,
    context: Any,
    model: Any,
    optimizer: Any,
    tokenizer: Any,
    config: Any,
    total_steps: int,
) -> dict[str, Any]:
    from ..tiny_v2.checkpoint import write_checkpoint
    from ..worker_context import CommitRejected
    from ..worker_protocol import WorkerProtocolError

    current_phase: str | None = None
    # The loading phase is emitted before these callbacks are built.
    # RUN-011 permits at most 32 phase events. Keep capacity for a terminal
    # checkpoint and the mandatory curriculum hold even when eval_every=1
    # would otherwise produce thousands of phase transitions.
    phase_event_count = 1

    def phase_changed(phase: str, *, terminal_checkpoint: bool = False) -> None:
        nonlocal current_phase, phase_event_count
        if phase == current_phase and not terminal_checkpoint:
            return
        current_phase = phase
        if phase == "cancellable_hold":
            limit = 32
        elif terminal_checkpoint:
            limit = 31
        else:
            limit = 30
        if phase_event_count >= limit:
            return
        context.emit("phase_changed", {"phase": phase})
        phase_event_count += 1

    def checkpoint(snapshot: Any) -> None:
        phase_changed(
            "checkpointing",
            terminal_checkpoint=snapshot.reason in {"final", "cancellation"},
        )
        name = f"checkpoint-{snapshot.state.completed_global_step}-{snapshot.checkpoint_id}"
        try:
            payload = write_checkpoint(
                context.staging_path / name,
                model=model,
                optimizer=optimizer,
                tokenizer=tokenizer,
                config=config,
                trainer_state=snapshot.state,
                batch_rng=trainer.batch_rng,
            )
            acknowledgment = context.commit_checkpoint(payload.proposal_dict())
        except (CommitRejected, WorkerProtocolError):
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            error = {
                "code": "CHECKPOINT_WRITE_FAILED",
                "message": "The Tiny-v2 checkpoint could not be committed.",
                "retryable": False,
                "field_errors": [],
            }
            if context.cancellation.cancelled:
                context.interrupt(
                    context.cancellation.reason or "user_cancelled", error
                )
            raise OperationFailure(
                "CHECKPOINT_WRITE_FAILED", error["message"], retryable=False
            ) from exc
        if (
            acknowledgment["checkpoint_id"] != snapshot.checkpoint_id
            or acknowledgment["step"] != snapshot.state.completed_global_step
        ):
            raise OperationFailure(
                "WORKER_PROTOCOL_ERROR",
                "The checkpoint acknowledgment does not match trainer state.",
            )

    def before_evaluation(step: int) -> None:
        phase_changed(
            "initial_evaluation" if step == 0 else "evaluation"
        )

    def before_step(step: int) -> None:
        phase_changed("training")

    def after_evaluation(value: Any) -> None:
        context.emit(
            "metric",
            {
                "phase": "validation",
                "step": value.step,
                "split": "validation",
                "name": "validation_nll_token",
                "value": value.validation_nll_token,
            },
        )

    def after_step(value: Any) -> None:
        context.emit(
            "progress",
            {
                "current": value.step,
                "total": total_steps,
                "unit": "updates",
                "message": "Training",
            },
        )

    def cancellation_reason() -> str | None:
        return (
            context.cancellation.reason
            if context.cancellation.cancelled
            else None
        )

    def enter_hold() -> str:
        phase_changed("cancellable_hold")
        if context.cancellation.wait(600.0):
            return context.cancellation.reason or "user_cancelled"
        return "exercise_timeout"

    # The callback closes over the trainer assigned immediately after return.
    trainer: Any = None

    def bind(value: Any) -> None:
        nonlocal trainer
        trainer = value

    return {
        "bind": bind,
        "checkpoint": checkpoint,
        "before_evaluation": before_evaluation,
        "before_step": before_step,
        "after_evaluation": after_evaluation,
        "after_step": after_step,
        "cancellation_reason": cancellation_reason,
        "enter_hold": enter_hold,
    }


def _finish(
    *,
    context: Any,
    operation: str,
    outcome: Any,
    parent_checkpoint_id: str | None = None,
) -> Mapping[str, Any]:
    run_id = str(context.output_allocations["run_id"])
    model_id = str(context.output_allocations["model_id"])
    metrics = _metrics_bytes(run_id, outcome.metrics)
    acknowledgment = context.stage_artifact(
        "training_metrics",
        metrics,
        filename=f"metrics-{run_id}.jsonl",
    )
    artifact_id = str(acknowledgment["artifact_id"])
    incumbent = outcome.state.incumbent
    if incumbent.checkpoint_id is None:
        raise OperationFailure(
            "WORKER_PROTOCOL_ERROR",
            "Tiny-v2 training ended without an incumbent checkpoint.",
        )
    result: dict[str, Any] = {
        "operation": operation,
        "run_id": run_id,
        "model_id": model_id,
        "completed_step": outcome.state.completed_global_step,
        "requested_final_step": outcome.state.requested_final_step,
        "artifact_ids": [artifact_id],
        "best_checkpoint_id": incumbent.checkpoint_id,
        "last_checkpoint_id": outcome.last_checkpoint_id,
        "metrics_artifact_id": artifact_id,
    }
    if parent_checkpoint_id is not None:
        result["parent_checkpoint_id"] = parent_checkpoint_id
    if outcome.interrupted:
        context.interrupt(outcome.interruption_reason or "user_cancelled")
    return result


def handle_tiny_train(
    request: Mapping[str, Any], context: Any
) -> Mapping[str, Any]:
    resolved = context.resolved
    device = str(context.snapshot["device"])
    seed = int(request["seed"])
    torch = _configure_runtime(seed, device)

    from ..tiny_v2.data import training_text_bytes
    from ..tiny_v2.model import Config, build_model
    from ..tiny_v2.tokenizer import load_tokenizer_bytes
    from ..tiny_v2.training import (
        TinyTrainer,
        TinyTrainingError,
        TrainingConfig,
        TrainingState,
        create_optimizer,
    )

    context.emit("phase_changed", {"phase": "loading"})
    train_documents = _read_dataset(context, "train")
    validation_documents = _read_dataset(context, "validation")
    if training_text_bytes(train_documents) > 200_000:
        raise OperationFailure(
            "WORKER_PROTOCOL_ERROR",
            "Verified Tiny-v2 training text exceeds its admitted byte limit.",
        )
    try:
        tokenizer = load_tokenizer_bytes(context.input("tokenizer_json").path.read_bytes())
    except (OSError, TypeError, ValueError) as exc:
        raise OperationFailure(
            "WORKER_PROTOCOL_ERROR", "The verified tokenizer artifact is invalid."
        ) from exc
    expected_tokenizer = resolved["tokenizer"]
    if (
        tokenizer.fingerprint() != expected_tokenizer["tokenizer_sha256"]
        or tokenizer.vocab_size != expected_tokenizer["vocab_size"]
    ):
        raise OperationFailure(
            "WORKER_PROTOCOL_ERROR", "The tokenizer snapshot identity changed."
        )

    config = Config(
        vocab_size=tokenizer.vocab_size,
        context=int(request["context"]),
        width=int(request["width"]),
        heads=int(request["heads"]),
        layers=int(request["layers"]),
    )
    train_data, validation_data = _encode_splits(
        train_documents, validation_documents, tokenizer, config.context
    )
    try:
        model = build_model(
            config,
            architecture_profile_id=str(request["architecture_profile_id"]),
        ).to(device)
    except torch.cuda.OutOfMemoryError as exc:
        raise OperationFailure(
            "OUT_OF_MEMORY", "Tiny-v2 training ran out of memory."
        ) from exc
    optimizer = create_optimizer(model, float(request["learning_rate"]))
    batch_rng = torch.Generator(device="cpu").manual_seed(seed)
    bindings = _dataset_bindings(resolved)
    training_config = TrainingConfig(
        requested_final_step=int(resolved["requested_final_step"]),
        eval_every=int(request["eval_every"]),
        batch_size=int(request["batch_size"]),
        seed=seed,
        learning_rate=float(request["learning_rate"]),
        architecture_profile_id=str(request["architecture_profile_id"]),
        runtime_profile=str(context.snapshot["runtime_profile"]),
        device=device,
        dependency_lock_sha256=str(context.snapshot["dependency_lock_sha256"]),
        dataset_bindings=bindings,
    )
    state = TrainingState.fresh(
        training_config,
        tokenizer_sha256=tokenizer.fingerprint(),
        parameter_names=dict(model.named_parameters(remove_duplicate=True)),
    )
    trainer = TinyTrainer(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        validation_data=validation_data,
        batch_rng=batch_rng,
        state=state,
        device=device,
    )
    callbacks = _make_callbacks(
        context=context,
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        config=config,
        total_steps=training_config.requested_final_step,
    )
    callbacks["bind"](trainer)
    try:
        outcome = trainer.run(
            next_checkpoint_id=context.checkpoint_identity,
            commit_checkpoint=callbacks["checkpoint"],
            cancellation_reason=callbacks["cancellation_reason"],
            after_step=callbacks["after_step"],
            before_step=callbacks["before_step"],
            after_evaluation=callbacks["after_evaluation"],
            before_evaluation=callbacks["before_evaluation"],
            hold_after_step=request.get("curriculum_hold_after_step"),
            enter_hold=callbacks["enter_hold"],
            evaluate_initial=True,
            checkpoint_initial=True,
        )
    except TinyTrainingError as exc:
        raise OperationFailure(exc.code, str(exc)) from exc
    except torch.cuda.OutOfMemoryError as exc:
        raise OperationFailure("OUT_OF_MEMORY", "Tiny-v2 training ran out of memory.") from exc
    except RuntimeError as exc:
        if "determin" in str(exc).lower():
            raise OperationFailure(
                "DETERMINISTIC_KERNEL_UNAVAILABLE",
                "A deterministic Tiny-v2 kernel is unavailable.",
            ) from exc
        raise
    return _finish(context=context, operation="tiny_train", outcome=outcome)


def handle_tiny_resume(
    request: Mapping[str, Any], context: Any
) -> Mapping[str, Any]:
    resolved = context.resolved
    parent = resolved["parent_checkpoint"]
    device = str(context.snapshot["device"])
    seed = int(parent["seed"])
    torch = _configure_runtime(seed, device)

    from ..tiny_v2.checkpoint import (
        CheckpointExpectations,
        CheckpointIncompatible,
        load_checkpoint,
    )
    from ..tiny_v2.data import training_text_bytes
    from ..tiny_v2.training import TinyTrainer, TinyTrainingError

    context.emit("phase_changed", {"phase": "loading"})
    bindings = _dataset_bindings(resolved)
    checkpoint_files = {
        name: context.input(f"checkpoint.{name}").path
        for name in (
            "model.safetensors",
            "optimizer.safetensors",
            "rng.safetensors",
            "tokenizer.json",
            "config.json",
            "trainer_state.json",
            "manifest.json",
        )
    }
    expected = CheckpointExpectations(
        manifest_sha256=str(parent["sha256"]),
        tokenizer_sha256=str(parent["model_identity"]["tokenizer_sha256"]),
        config_sha256=str(parent["model_identity"]["config_sha256"]),
        architecture_profile_id=str(
            parent["model_identity"]["architecture_profile_id"]
        ),
        dataset_bindings=bindings,
        runtime_profile=str(context.snapshot["runtime_profile"]),
        device=device,
        dependency_lock_sha256=str(context.snapshot["dependency_lock_sha256"]),
        completed_global_step=int(parent["step"]),
    )
    try:
        loaded = load_checkpoint(
            checkpoint_files,
            device=device,
            expected=expected,
            for_resume=True,
        )
    except CheckpointIncompatible as exc:
        raise OperationFailure("CHECKPOINT_INCOMPATIBLE", str(exc)) from exc
    except torch.cuda.OutOfMemoryError as exc:
        raise OperationFailure(
            "OUT_OF_MEMORY", "Tiny-v2 training ran out of memory."
        ) from exc
    assert loaded.trainer_state is not None
    assert loaded.optimizer is not None
    assert loaded.batch_rng is not None

    train_documents = _read_dataset(context, "train")
    validation_documents = _read_dataset(context, "validation")
    if training_text_bytes(train_documents) > 200_000:
        raise OperationFailure(
            "WORKER_PROTOCOL_ERROR",
            "Verified Tiny-v2 training text exceeds its admitted byte limit.",
        )
    train_data, validation_data = _encode_splits(
        train_documents,
        validation_documents,
        loaded.tokenizer,
        loaded.config.context,
    )
    state = replace(
        loaded.trainer_state,
        requested_final_step=int(resolved["requested_final_step"]),
    )
    trainer = TinyTrainer(
        model=loaded.model,
        optimizer=loaded.optimizer,
        train_data=train_data,
        validation_data=validation_data,
        batch_rng=loaded.batch_rng,
        state=state,
        device=device,
        last_checkpoint_id=str(request["checkpoint_id"]),
        last_checkpoint_step=state.completed_global_step,
    )
    callbacks = _make_callbacks(
        context=context,
        model=loaded.model,
        optimizer=loaded.optimizer,
        tokenizer=loaded.tokenizer,
        config=loaded.config,
        total_steps=state.requested_final_step,
    )
    callbacks["bind"](trainer)
    try:
        outcome = trainer.run(
            next_checkpoint_id=context.checkpoint_identity,
            commit_checkpoint=callbacks["checkpoint"],
            cancellation_reason=callbacks["cancellation_reason"],
            after_step=callbacks["after_step"],
            before_step=callbacks["before_step"],
            after_evaluation=callbacks["after_evaluation"],
            before_evaluation=callbacks["before_evaluation"],
            evaluate_initial=True,
            checkpoint_initial=False,
        )
    except TinyTrainingError as exc:
        raise OperationFailure(exc.code, str(exc)) from exc
    except torch.cuda.OutOfMemoryError as exc:
        raise OperationFailure("OUT_OF_MEMORY", "Tiny-v2 training ran out of memory.") from exc
    except RuntimeError as exc:
        if "determin" in str(exc).lower():
            raise OperationFailure(
                "DETERMINISTIC_KERNEL_UNAVAILABLE",
                "A deterministic Tiny-v2 kernel is unavailable.",
            ) from exc
        raise
    return _finish(
        context=context,
        operation="tiny_resume",
        outcome=outcome,
        parent_checkpoint_id=str(request["checkpoint_id"]),
    )


__all__ = ["handle_tiny_resume", "handle_tiny_train"]
