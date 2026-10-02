"""Support coverage for reachable Tiny-v2 model failure boundaries."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import torch

# Isolated-mode direct execution omits the script directory from sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _tiny_fixtures import (
    canonical_json,
    clone_checkpoint,
    create_resume_checkpoint,
    refresh_manifest_entry,
    rewrite_json_payload,
    rewrite_safetensors,
)
from llm_foundations_companion.tiny_v2.checkpoint import (
    CheckpointExpectations,
    CheckpointIncompatible,
    load_checkpoint,
    load_config_bytes,
    materialize_model,
    validate_checkpoint,
)
from llm_foundations_companion.operations.tiny_training import (
    _make_callbacks,
    _metrics_bytes,
)
from llm_foundations_companion.schema import strict_json, validate
from llm_foundations_companion.tiny_v2.decoding import (
    generate_tokens,
    select_next_token,
    top_p_distribution,
)
from llm_foundations_companion.tiny_v2.evaluation import (
    EvaluationCancelled,
    aggregate_evaluation_rows,
    evaluate_nll_per_byte,
)
from llm_foundations_companion.tiny_v2.model import (
    Config,
    STANDARD_PROFILE,
    build_model,
)
from llm_foundations_companion.tiny_v2.tokenizer import Tokenizer
from llm_foundations_companion.tiny_v2.training import (
    DatasetBinding,
    Evaluation,
    Incumbent,
    StepMetric,
    TinyTrainer,
    TinyTrainingError,
    TrainingConfig,
    TrainingState,
    create_optimizer,
    evaluate_fixed,
)
from llm_foundations_companion.worker_protocol import PROTOCOL_VERSION, ProtocolState


def _bindings() -> tuple[DatasetBinding, DatasetBinding]:
    return (
        DatasetBinding(
            "train",
            "71000000-0000-4000-8000-000000000001",
            "1" * 64,
        ),
        DatasetBinding(
            "validation",
            "71000000-0000-4000-8000-000000000002",
            "2" * 64,
        ),
    )


def _config_values(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "requested_final_step": 2,
        "eval_every": 1,
        "batch_size": 1,
        "seed": 17,
        "learning_rate": 0.001,
        "architecture_profile_id": STANDARD_PROFILE,
        "runtime_profile": "wsl-cpu",
        "device": "cpu",
        "dependency_lock_sha256": "3" * 64,
        "dataset_bindings": _bindings(),
    }
    values.update(changes)
    return values


def _training_state(
    parameter_names: tuple[str, ...] = ("weight",),
    **config_changes: object,
) -> TrainingState:
    config = TrainingConfig(**_config_values(**config_changes))  # type: ignore[arg-type]
    return TrainingState.fresh(
        config,
        tokenizer_sha256="4" * 64,
        parameter_names=parameter_names,
    )


class _LossModel(torch.nn.Module):
    def __init__(self, *, nonfinite_loss: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.cfg = SimpleNamespace(context=1)
        self.nonfinite_loss = nonfinite_loss

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del inputs, targets
        value = float("nan") if self.nonfinite_loss else 1.0
        loss = self.weight.square() * value
        return torch.zeros((1, 1, 257)), loss


class _LogitModel(torch.nn.Module):
    def __init__(self, outputs: list[int] | None = None) -> None:
        super().__init__()
        self.outputs = list(outputs or [0])
        self.calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        token_id = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        logits = torch.full(
            (inputs.shape[0], inputs.shape[1], 257),
            -100.0,
            dtype=torch.float64,
        )
        logits[:, -1, token_id] = 100.0
        return logits


class S2ModelFailureCoverage(unittest.TestCase):
    def test_training_callback_emits_all_steps_as_valid_protocol_progress(
        self,
    ) -> None:
        job_id = "70000000-0000-4000-8000-000000000010"
        instance_id = "70000000-0000-4000-8000-000000000011"
        spawn_nonce = "a" * 64
        request_sha256 = "b" * 64
        input_snapshot_sha256 = "c" * 64
        schema_id = "TinyTrainRequest"
        state = ProtocolState(
            job_id,
            instance_id,
            spawn_nonce,
            request_sha256,
            schema_id,
            "tiny_train",
            input_snapshot_sha256,
        )
        state.accept(
            {
                "type": "ready",
                "protocol_version": PROTOCOL_VERSION,
                "job_id": job_id,
                "instance_id": instance_id,
                "spawn_nonce": spawn_nonce,
                "request_sha256": request_sha256,
                "schema_id": schema_id,
                "input_snapshot_sha256": input_snapshot_sha256,
            }
        )

        accepted: list[dict[str, Any]] = []

        def emit(event_type: str, payload: dict[str, Any]) -> None:
            envelope = state.accept(
                {"type": "event", "event_type": event_type, "payload": payload}
            )
            accepted.append(dict(envelope))

        context = SimpleNamespace(
            emit=emit,
            cancellation=SimpleNamespace(cancelled=False, reason=None),
            staging_path=Path("."),
        )
        callbacks = _make_callbacks(
            context=context,
            model=None,
            optimizer=None,
            tokenizer=None,
            config=None,
            total_steps=50,
        )
        for step in range(1, 51):
            callbacks["after_step"](SimpleNamespace(step=step))

        self.assertEqual(len(accepted), 50)
        self.assertEqual(
            [message["payload"]["current"] for message in accepted],
            list(range(1, 51)),
        )
        self.assertEqual(
            accepted[-1],
            {
                "type": "event",
                "event_type": "progress",
                "payload": {
                    "current": 50,
                    "total": 50,
                    "unit": "updates",
                    "message": "Training",
                },
            },
        )

    def test_training_metric_serializer_matches_sealed_long_form_schema(
        self,
    ) -> None:
        run_id = "70000000-0000-4000-8000-000000000001"
        raw = _metrics_bytes(
            run_id,
            (
                Evaluation(0, 2.0, 3.0),
                StepMetric(1, 1.5, 0.5, 0.25),
                Evaluation(1, 1.25, 2.5),
            ),
        )
        self.assertTrue(raw.endswith(b"\n"))
        rows = [
            strict_json(line)
            for line in raw[:-1].split(b"\n")
        ]
        for row in rows:
            validate("metric-record.schema.json", row)
            self.assertEqual(
                set(row),
                {
                    "run_id",
                    "sequence",
                    "step",
                    "name",
                    "value",
                    "unit",
                    "protocol_id",
                    "recorded_at",
                },
            )
        self.assertEqual([row["sequence"] for row in rows], list(range(7)))
        self.assertEqual([row["step"] for row in rows], [0, 0, 1, 1, 1, 1, 1])
        self.assertEqual(
            [row["name"] for row in rows],
            [
                "train_nll_token",
                "validation_nll_token",
                "train_nll_token",
                "gradient_l2_norm",
                "elapsed_seconds",
                "train_nll_token",
                "validation_nll_token",
            ],
        )
        self.assertEqual(
            [row["unit"] for row in rows],
            [
                "nats_per_token",
                "nats_per_token",
                "nats_per_token",
                "l2_norm",
                "seconds",
                "nats_per_token",
                "nats_per_token",
            ],
        )
        self.assertEqual(
            [row["protocol_id"] for row in rows],
            [
                "tiny-v2-fixed-eight-batches-v1",
                "tiny-v2-fixed-eight-batches-v1",
                "tiny-v2-training-v1",
                "tiny-v2-training-v1",
                "tiny-v2-training-v1",
                "tiny-v2-fixed-eight-batches-v1",
                "tiny-v2-fixed-eight-batches-v1",
            ],
        )
        recorded_at = {row["recorded_at"] for row in rows}
        self.assertEqual(len(recorded_at), 1)
        self.assertTrue(str(next(iter(recorded_at))).endswith("Z"))

    def test_checkpoint_config_and_source_boundaries(self) -> None:
        valid = {
            "format": "tiny-v2-config-v1",
            "architecture_profile_id": STANDARD_PROFILE,
            "vocab_size": 257,
            "context": 8,
            "width": 16,
            "heads": 1,
            "layers": 1,
        }
        parsed, architecture, digest = load_config_bytes(canonical_json(valid))
        self.assertEqual(parsed, Config(257, 8, 16, 1, 1))
        self.assertEqual(architecture, STANDARD_PROFILE)
        self.assertEqual(len(digest), 64)

        malformed = [
            b"\xff",
            b"[]",
            b'{"format":"x","format":"y"}',
            b'{"format":NaN}',
            b"{",
        ]
        mutations: list[dict[str, object]] = []
        for field, value in (
            ("format", "other"),
            ("architecture_profile_id", "other"),
            ("context", True),
            ("context", 7),
            ("heads", 3),
        ):
            candidate = dict(valid)
            candidate[field] = value
            mutations.append(candidate)
        missing = dict(valid)
        missing.pop("layers")
        mutations.append(missing)
        capped = dict(valid)
        capped.update(
            vocab_size=1024,
            context=512,
            width=512,
            heads=16,
            layers=12,
        )
        mutations.append(capped)
        for raw in [*malformed, *(canonical_json(item) for item in mutations)]:
            with self.subTest(raw=raw[:80]), self.assertRaises(
                CheckpointIncompatible
            ):
                load_config_bytes(raw)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(CheckpointIncompatible, "not a directory"):
                validate_checkpoint(root / "missing")
            with self.assertRaisesRegex(CheckpointIncompatible, "no manifest"):
                validate_checkpoint({})
            manifest = root / "manifest.json"
            manifest.write_bytes(b"{}")
            with self.assertRaisesRegex(CheckpointIncompatible, "unsafe"):
                validate_checkpoint(
                    {"manifest.json": manifest, "unexpected.bin": manifest}
                )
            with self.assertRaisesRegex(CheckpointIncompatible, "missing file"):
                validate_checkpoint(
                    {"manifest.json": root / "does-not-exist"}
                )

    def test_checkpoint_expectations_and_tensor_failures(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = create_resume_checkpoint(root)
            proposal = fixture.payload.proposal_dict()
            self.assertEqual(proposal["staging_name"], fixture.directory.name)
            self.assertEqual(len(proposal["files"]), 7)

            mismatches = (
                CheckpointExpectations(manifest_sha256="0" * 64),
                CheckpointExpectations(tokenizer_sha256="0" * 64),
                CheckpointExpectations(config_sha256="0" * 64),
                CheckpointExpectations(
                    architecture_profile_id="tiny-v2-weight-tied-v1"
                ),
                CheckpointExpectations(runtime_profile="win-cpu"),
                CheckpointExpectations(device="cuda"),
                CheckpointExpectations(dependency_lock_sha256="0" * 64),
                CheckpointExpectations(completed_global_step=0),
                CheckpointExpectations(
                    dataset_bindings=(
                        DatasetBinding(
                            "train",
                            "72000000-0000-4000-8000-000000000001",
                            "a" * 64,
                        ),
                        DatasetBinding(
                            "validation",
                            "72000000-0000-4000-8000-000000000002",
                            "b" * 64,
                        ),
                    )
                ),
            )
            for expected in mismatches:
                with self.subTest(expected=expected), self.assertRaises(
                    CheckpointIncompatible
                ):
                    validate_checkpoint(fixture.directory, expected=expected)

            with self.assertRaisesRegex(CheckpointIncompatible, "resume device"):
                load_checkpoint(
                    fixture.directory,
                    device="cuda",
                    for_resume=True,
                )

            validated = validate_checkpoint(fixture.directory, for_resume=True)
            with self.assertRaisesRegex(CheckpointIncompatible, "materialized"):
                materialize_model(
                    replace(validated, model_tensors={}),
                    device="cpu",
                )

            def nonfinite(tensors: dict[str, torch.Tensor]) -> None:
                name = next(
                    key
                    for key, tensor in tensors.items()
                    if tensor.is_floating_point()
                )
                value = tensors[name].clone()
                value.reshape(-1)[0] = math.nan
                tensors[name] = value

            def missing_rng(tensors: dict[str, torch.Tensor]) -> None:
                tensors.pop("batch_cpu")

            def wrong_rng_dtype(tensors: dict[str, torch.Tensor]) -> None:
                tensors["batch_cpu"] = tensors["batch_cpu"].to(torch.int64)

            scenarios: dict[str, object] = {}
            corrupt = clone_checkpoint(
                fixture.directory, root / "corrupt-safetensors"
            )
            (corrupt / "model.safetensors").write_bytes(b"not-safetensors")
            refresh_manifest_entry(corrupt, "model.safetensors")
            scenarios["corrupt_model_encoding"] = corrupt
            for label, name, mutation in (
                ("nonfinite_model", "model.safetensors", nonfinite),
                ("nonfinite_optimizer", "optimizer.safetensors", nonfinite),
                ("missing_rng", "rng.safetensors", missing_rng),
                ("wrong_rng_dtype", "rng.safetensors", wrong_rng_dtype),
            ):
                candidate = clone_checkpoint(
                    fixture.directory, root / label
                )
                rewrite_safetensors(candidate, name, mutation)
                scenarios[label] = candidate

            stale_steps = clone_checkpoint(
                fixture.directory, root / "stale-parameter-step"
            )

            def change_step(value: dict[str, object]) -> None:
                steps = value["parameter_steps"]
                assert isinstance(steps, dict)
                first = next(iter(steps))
                steps[first] = 0

            rewrite_json_payload(stale_steps, "trainer_state.json", change_step)
            scenarios["stale_parameter_step"] = stale_steps

            for label, candidate in scenarios.items():
                with self.subTest(label=label), self.assertRaises(
                    CheckpointIncompatible
                ):
                    validate_checkpoint(candidate, for_resume=True)

    def test_training_configuration_and_restored_state_bounds(self) -> None:
        for changes in (
            {"requested_final_step": 0},
            {"eval_every": 501},
            {"batch_size": 65},
            {"seed": -1},
            {"learning_rate": 0.0},
            {"learning_rate": math.inf},
            {"architecture_profile_id": "other"},
            {"runtime_profile": "other"},
            {"runtime_profile": "wsl-cuda", "device": "cpu"},
            {"dependency_lock_sha256": "short"},
            {"dataset_bindings": (_bindings()[0],)},
            {"dataset_bindings": (_bindings()[0], _bindings()[0])},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                TrainingConfig(**_config_values(**changes))  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            DatasetBinding("test", "id", "1" * 64)
        with self.assertRaises(ValueError):
            DatasetBinding("train", "", "1" * 64)
        for incumbent in (
            {"checkpoint_id": "id"},
            {
                "checkpoint_id": "id",
                "validation_nll_token": math.inf,
                "step": 0,
            },
            {
                "checkpoint_id": "id",
                "validation_nll_token": 1.0,
                "step": -1,
            },
        ):
            with self.subTest(incumbent=incumbent), self.assertRaises(ValueError):
                Incumbent(**incumbent)  # type: ignore[arg-type]

        state = _training_state()
        invalid_state_changes = (
            {"completed_global_step": 3},
            {"tokenizer_sha256": "short"},
            {"parameter_steps": {}},
            {"parameter_steps": {"weight": -1}},
            {
                "completed_global_step": 2_147_483_648,
                "requested_final_step": 2_147_483_648,
            },
            {"incumbent": Incumbent("checkpoint", 1.0, 1)},
            {"architecture_profile_id": "other"},
            {"runtime_profile": "other"},
            {"runtime_profile": "wsl-cuda"},
        )
        for changes in invalid_state_changes:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(state, **changes)

        with self.assertRaises(ValueError):
            Evaluation(-1, 1.0, 1.0)
        for losses in ((math.inf, 1.0), (1.0, math.nan)):
            with self.assertRaises(TinyTrainingError) as caught:
                Evaluation(0, *losses)
            self.assertEqual(caught.exception.code, "NONFINITE_TRAINING_VALUE")

    def test_trainer_rejects_nondurable_and_nonfinite_boundaries(self) -> None:
        model = _LossModel()
        state = _training_state(("weight",))
        optimizer = create_optimizer(model, state.learning_rate)
        data = torch.arange(8, dtype=torch.long)
        rng = torch.Generator(device="cpu").manual_seed(17)

        def trainer(
            current: TrainingState = state,
            **checkpoint: object,
        ) -> TinyTrainer:
            return TinyTrainer(
                model=model,
                optimizer=optimizer,
                train_data=data,
                validation_data=data,
                batch_rng=rng,
                state=current,
                device="cpu",
                **checkpoint,
            )

        with self.assertRaises(ValueError):
            trainer(last_checkpoint_id="id")
        with self.assertRaises(ValueError):
            trainer(last_checkpoint_id="id", last_checkpoint_step=1)

        active = trainer()
        stale = active.checkpoint_snapshot(
            "73000000-0000-4000-8000-000000000001",
            reason="initial",
            evaluation=None,
        )
        active.state = replace(
            active.state,
            completed_global_step=1,
            parameter_steps={"weight": 1},
        )
        with self.assertRaisesRegex(RuntimeError, "stale trainer state"):
            active.accept_checkpoint(stale)

        complete = replace(
            state,
            completed_global_step=state.requested_final_step,
            parameter_steps={"weight": state.requested_final_step},
        )
        with self.assertRaisesRegex(RuntimeError, "durable checkpoint"):
            trainer(complete).run(
                next_checkpoint_id=lambda: "unused",
                commit_checkpoint=lambda _snapshot: None,
                cancellation_reason=lambda: None,
                evaluate_initial=False,
            )

        batch_value = (
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1, 1), dtype=torch.long),
        )
        nonfinite_model = _LossModel(nonfinite_loss=True)
        nonfinite_trainer = TinyTrainer(
            model=nonfinite_model,
            optimizer=create_optimizer(nonfinite_model, 0.001),
            train_data=data,
            validation_data=data,
            batch_rng=rng,
            state=_training_state(("weight",)),
            device="cpu",
        )
        with patch(
            "llm_foundations_companion.tiny_v2.training.batch",
            return_value=batch_value,
        ):
            with self.assertRaises(TinyTrainingError) as loss_error:
                nonfinite_trainer.step_once()
        self.assertEqual(loss_error.exception.code, "NONFINITE_TRAINING_VALUE")

        finite_trainer = trainer()
        with patch(
            "llm_foundations_companion.tiny_v2.training.batch",
            return_value=batch_value,
        ), patch(
            "torch.nn.utils.clip_grad_norm_",
            side_effect=RuntimeError("forced nonfinite gradient"),
        ):
            with self.assertRaises(TinyTrainingError) as gradient_error:
                finite_trainer.step_once()
        self.assertEqual(
            gradient_error.exception.code, "NONFINITE_TRAINING_VALUE"
        )

        finite_trainer = trainer()
        with patch(
            "llm_foundations_companion.tiny_v2.training.batch",
            return_value=batch_value,
        ), patch(
            "torch.nn.utils.clip_grad_norm_",
            return_value=torch.tensor(math.nan),
        ):
            with self.assertRaises(TinyTrainingError):
                finite_trainer.step_once()

        was_training = nonfinite_model.training
        with patch(
            "llm_foundations_companion.tiny_v2.training.batch",
            return_value=batch_value,
        ):
            with self.assertRaises(TinyTrainingError):
                evaluate_fixed(
                    nonfinite_model,
                    data,
                    batch_size=1,
                    seed=17,
                    device="cpu",
                    batches=1,
                )
        self.assertEqual(nonfinite_model.training, was_training)

    def test_trainer_hold_contract_requires_a_committed_boundary(self) -> None:
        model = _LossModel()
        data = torch.arange(8, dtype=torch.long)

        def make_trainer(*, final_step: int, eval_every: int) -> TinyTrainer:
            state = _training_state(
                ("weight",),
                requested_final_step=final_step,
                eval_every=eval_every,
            )
            return TinyTrainer(
                model=model,
                optimizer=create_optimizer(model, 0.001),
                train_data=data,
                validation_data=data,
                batch_rng=torch.Generator(device="cpu").manual_seed(9),
                state=state,
                device="cpu",
            )

        def step_once(target: TinyTrainer) -> StepMetric:
            step = target.state.completed_global_step + 1
            target.state = replace(
                target.state,
                completed_global_step=step,
                parameter_steps={"weight": step},
            )
            return StepMetric(step, 1.0, 1.0, 0.0)

        uncommitted = make_trainer(final_step=2, eval_every=2)
        with patch.object(
            uncommitted, "step_once", side_effect=lambda: step_once(uncommitted)
        ):
            with self.assertRaisesRegex(RuntimeError, "coincide"):
                uncommitted.run(
                    next_checkpoint_id=lambda: "unused",
                    commit_checkpoint=lambda _snapshot: None,
                    cancellation_reason=lambda: None,
                    hold_after_step=1,
                    enter_hold=lambda: None,
                    evaluate_initial=False,
                )

        committed = make_trainer(final_step=1, eval_every=1)
        evaluation = Evaluation(1, 1.0, 1.0)
        with patch.object(
            committed, "step_once", side_effect=lambda: step_once(committed)
        ), patch.object(committed, "evaluate", return_value=evaluation):
            with self.assertRaisesRegex(RuntimeError, "scheduler callback"):
                committed.run(
                    next_checkpoint_id=lambda: (
                        "73000000-0000-4000-8000-000000000002"
                    ),
                    commit_checkpoint=lambda _snapshot: None,
                    cancellation_reason=lambda: None,
                    hold_after_step=1,
                    evaluate_initial=False,
                )

    def test_evaluation_rejects_corrupt_rows_and_model_outputs(self) -> None:
        invalid_rows = (
            (),
            ({"negative_log_likelihood": True, "utf8_bytes": 1},),
            ({"negative_log_likelihood": -1.0, "utf8_bytes": 1},),
            ({"negative_log_likelihood": 1.0, "utf8_bytes": True},),
            ({"negative_log_likelihood": 1.0, "utf8_bytes": 0},),
        )
        for rows in invalid_rows:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                aggregate_evaluation_rows(rows)

        tokenizer = Tokenizer()

        class WrongRank(torch.nn.Module):
            def forward(self, _inputs: torch.Tensor) -> torch.Tensor:
                return torch.zeros((1, 257))

        class Nonfinite(torch.nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return torch.full(
                    (inputs.shape[0], inputs.shape[1], 257), math.inf
                )

        record = ({"record_id": "record-a", "text": "a"},)
        for model in (WrongRank(), Nonfinite()):
            with self.subTest(model=type(model).__name__), self.assertRaises(
                ValueError
            ):
                evaluate_nll_per_byte(
                    model,
                    tokenizer,
                    record,
                    context=8,
                    device="cpu",
                )

        with self.assertRaises(EvaluationCancelled):
            evaluate_nll_per_byte(
                _LogitModel([0]),
                tokenizer,
                record,
                context=8,
                device="cpu",
                cancelled=lambda: True,
            )

    def test_decoding_rejects_nonfinite_logits_and_preserves_stop_causes(
        self,
    ) -> None:
        for logits in (
            torch.tensor([]),
            torch.zeros((1, 2)),
            torch.tensor([0.0, math.nan]),
        ):
            with self.subTest(shape=tuple(logits.shape)), self.assertRaises(
                ValueError
            ):
                top_p_distribution(logits, temperature=1.0, top_p=1.0)

        with self.assertRaisesRegex(ValueError, "top_p must equal 1"):
            select_next_token(
                torch.tensor([0.0, 1.0]),
                temperature=0.0,
                top_p=0.9,
            )
        with self.assertRaisesRegex(ValueError, "job-local generator"):
            select_next_token(
                torch.tensor([0.0, 1.0]),
                temperature=1.0,
                top_p=1.0,
            )
        self.assertEqual(
            select_next_token(
                torch.tensor([1.0, 1.0, 0.0]),
                temperature=0.0,
                top_p=1.0,
            ),
            0,
        )
        distribution = top_p_distribution(
            torch.tensor([0.0, 0.0, -100.0]),
            temperature=1.0,
            top_p=0.6,
        )
        self.assertEqual(distribution.token_ids, (0, 1))
        self.assertAlmostEqual(
            float(distribution.probabilities.sum().item()), 1.0
        )

        tokenizer = Tokenizer()

        class MustNotRun(torch.nn.Module):
            def forward(self, _inputs: torch.Tensor) -> torch.Tensor:
                raise AssertionError("cancelled generation invoked the model")

        cancelled = generate_tokens(
            MustNotRun(),
            tokenizer,
            "prompt",
            context=8,
            max_new_tokens=1,
            temperature=0.0,
            top_p=1.0,
            seed=1,
            device="cpu",
            cancelled=lambda: True,
        )
        self.assertEqual(cancelled.stop_reason, "cancelled")
        self.assertEqual(cancelled.generated_token_count, 0)

        eos = generate_tokens(
            _LogitModel([tokenizer.eos_id]),
            tokenizer,
            "prompt",
            context=8,
            max_new_tokens=2,
            temperature=0.0,
            top_p=1.0,
            seed=1,
            device="cpu",
        )
        self.assertEqual(eos.stop_reason, "eos")
        self.assertEqual(eos.generated_text, "")

        replacement = generate_tokens(
            _LogitModel([0xF0, tokenizer.eos_id]),
            tokenizer,
            "prompt",
            context=8,
            max_new_tokens=2,
            temperature=0.0,
            top_p=1.0,
            seed=1,
            device="cpu",
        )
        self.assertTrue(replacement.replacement_occurred)
        self.assertEqual(replacement.generated_text, "\ufffd")

        class WrongRank(torch.nn.Module):
            def forward(self, _inputs: torch.Tensor) -> torch.Tensor:
                return torch.zeros((1, 257))

        with self.assertRaises(ValueError):
            generate_tokens(
                WrongRank(),
                tokenizer,
                "prompt",
                context=8,
                max_new_tokens=1,
                temperature=0.0,
                top_p=1.0,
                seed=1,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
