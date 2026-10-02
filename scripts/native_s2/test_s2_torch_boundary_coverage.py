"""Focused source-runtime coverage for Tiny-v2 Torch state boundaries."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import os
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
from llm_foundations_companion.operations import (
    OperationFailure,
    OperationUnavailable,
    handler_for,
)
from llm_foundations_companion.tiny_v2.checkpoint import (
    CheckpointIncompatible,
    load_checkpoint,
    validate_checkpoint,
    write_checkpoint,
)
from llm_foundations_companion.tiny_v2.determinism import (
    configure_torch_determinism,
)
from llm_foundations_companion.tiny_v2.metadata import load_config_metadata
from llm_foundations_companion.tiny_v2.model import (
    Config,
    STANDARD_PROFILE,
    TIED_PROFILE,
    build_model,
)
from llm_foundations_companion.tiny_v2.tokenizer import Tokenizer
from llm_foundations_companion.tiny_v2.training import (
    CheckpointSnapshot,
    DatasetBinding,
    TinyTrainer,
    TrainingConfig,
    TrainingState,
    create_optimizer,
)


def _trainer(*, seed: int = 41) -> TinyTrainer:
    torch.manual_seed(seed)
    tokenizer = Tokenizer()
    config = Config(tokenizer.vocab_size, 8, 16, 1, 1)
    model = build_model(config, STANDARD_PROFILE)
    optimizer = create_optimizer(model, 0.001)
    training_config = TrainingConfig(
        requested_final_step=2,
        eval_every=2,
        batch_size=2,
        seed=seed,
        learning_rate=0.001,
        architecture_profile_id=STANDARD_PROFILE,
        runtime_profile="wsl-cpu",
        device="cpu",
        dependency_lock_sha256="d" * 64,
        dataset_bindings=(
            DatasetBinding(
                "train",
                "41000000-0000-4000-8000-000000000001",
                "1" * 64,
            ),
            DatasetBinding(
                "validation",
                "41000000-0000-4000-8000-000000000002",
                "2" * 64,
            ),
        ),
    )
    state = TrainingState.fresh(
        training_config,
        tokenizer_sha256=tokenizer.fingerprint(),
        parameter_names=dict(model.named_parameters(remove_duplicate=True)),
    )
    data = torch.arange(96, dtype=torch.long) % tokenizer.vocab_size
    return TinyTrainer(
        model=model,
        optimizer=optimizer,
        train_data=data,
        validation_data=data.flip(0),
        batch_rng=torch.Generator(device="cpu").manual_seed(seed),
        state=state,
        device="cpu",
    )


class _Interrupted(RuntimeError):
    pass


class _OperationContext:
    _artifact_ids = {
        "evaluation_records": "61000000-0000-4000-8000-000000000001",
        "evaluation_metrics": "61000000-0000-4000-8000-000000000002",
        "evaluation_paired": "61000000-0000-4000-8000-000000000003",
        "context_preview": "61000000-0000-4000-8000-000000000004",
        "generation": "61000000-0000-4000-8000-000000000005",
    }

    def __init__(
        self,
        *,
        inputs: dict[str, SimpleNamespace],
        resolved: dict[str, object],
        run_id: str | None,
    ) -> None:
        self.job_id = "61000000-0000-4000-8000-000000000010"
        self.runtime_profile = "wsl-cpu"
        self.snapshot = {
            "device": "cpu",
            "inputs": [{"role": role} for role in inputs],
        }
        self.resolved = resolved
        self.output_allocations = {"run_id": run_id}
        self.cancellation = SimpleNamespace(cancelled=False, reason=None)
        self._inputs = inputs
        self.artifacts: dict[str, bytes] = {}
        self.events: list[tuple[str, dict[str, object]]] = []

    def input(self, role: str) -> SimpleNamespace:
        return self._inputs[role]

    def emit(self, event_type: str, payload: dict[str, object]) -> None:
        self.events.append((event_type, payload))

    def stage_artifact(
        self, role: str, data: bytes, *, filename: str
    ) -> dict[str, object]:
        del filename
        self.artifacts[role] = data
        return {"artifact_id": self._artifact_ids[role]}

    def interrupt(self, reason: str) -> None:
        raise _Interrupted(reason)


def _checkpoint_descriptor(fixture: Any) -> dict[str, object]:
    return {
        "model_id": "62000000-0000-4000-8000-000000000001",
        "checkpoint_id": "62000000-0000-4000-8000-000000000002",
        "checkpoint_sha256": fixture.payload.manifest_sha256,
        "manifest_sha256": fixture.payload.manifest_sha256,
        "tokenizer_sha256": fixture.tokenizer.fingerprint(),
        "config_sha256": fixture.payload.config_sha256,
        "architecture_profile_id": STANDARD_PROFILE,
    }


def _checkpoint_inputs(fixture: Any, prefix: str) -> dict[str, SimpleNamespace]:
    return {
        f"{prefix}{path.name}": SimpleNamespace(
            path=path,
            artifact_id=f"input-{prefix}{path.name}",
        )
        for path in fixture.directory.iterdir()
    }


class S2TorchBoundaryCoverage(unittest.TestCase):
    def test_config_metadata_accepts_profiles_and_rejects_malformed_values(
        self,
    ) -> None:
        base: dict[str, object] = {
            "format": "tiny-v2-config-v1",
            "architecture_profile_id": STANDARD_PROFILE,
            "vocab_size": 257,
            "context": 8,
            "width": 16,
            "heads": 1,
            "layers": 1,
        }
        expected_config = {
            "vocab_size": 257,
            "context": 8,
            "width": 16,
            "heads": 1,
            "layers": 1,
        }
        for architecture in (STANDARD_PROFILE, TIED_PROFILE):
            with self.subTest(architecture=architecture):
                value = {**base, "architecture_profile_id": architecture}
                raw = canonical_json(value)
                config, loaded_architecture, digest = load_config_metadata(raw)
                self.assertEqual(config, expected_config)
                self.assertEqual(loaded_architecture, architecture)
                self.assertEqual(digest, hashlib.sha256(raw).hexdigest())

        with self.assertRaisesRegex(TypeError, "must be bytes"):
            load_config_metadata("{}")  # type: ignore[arg-type]

        malformed = {
            "invalid_utf8": (b"\xff", "not strict UTF-8 JSON"),
            "invalid_json": (b'{"format":', "not strict UTF-8 JSON"),
            "non_object": (b"[]", "must contain a JSON object"),
            "duplicate_key": (
                b'{"format":"first","format":"second"}',
                "duplicate JSON key",
            ),
            "nonfinite": (b'{"value":NaN}', "nonfinite number"),
        }
        for label, (raw, message) in malformed.items():
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError, message
            ):
                load_config_metadata(raw)

        invalid_objects = {
            "missing_required": (
                {key: value for key, value in base.items() if key != "layers"},
                "invalid fields or format",
            ),
            "unknown_optional": (
                {**base, "optional": 1},
                "invalid fields or format",
            ),
            "wrong_format": (
                {**base, "format": "tiny-v2-config-v0"},
                "invalid fields or format",
            ),
            "unknown_architecture": (
                {**base, "architecture_profile_id": "unknown"},
                "unknown architecture profile",
            ),
            "boolean_integer": (
                {**base, "context": True},
                "field context is outside its bounds",
            ),
            "vocab_too_small": (
                {**base, "vocab_size": 256},
                "field vocab_size is outside its bounds",
            ),
            "width_too_large": (
                {**base, "width": 513},
                "field width is outside its bounds",
            ),
            "heads_too_small": (
                {**base, "heads": 0},
                "field heads is outside its bounds",
            ),
            "layers_too_large": (
                {**base, "layers": 13},
                "field layers is outside its bounds",
            ),
            "width_not_divisible": (
                {**base, "width": 17, "heads": 2},
                "width must divide evenly by heads",
            ),
        }
        for label, (value, message) in invalid_objects.items():
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError, message
            ):
                load_config_metadata(canonical_json(value))

    def test_determinism_rejects_invalid_inputs_and_unavailable_cuda(self) -> None:
        for seed in (True, -1, 4_294_967_296, 1.5):
            with self.subTest(seed=seed), self.assertRaisesRegex(
                ValueError, "seed must be an integer"
            ):
                configure_torch_determinism(seed, "cpu")  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "device must be cpu or cuda"):
            configure_torch_determinism(0, "gpu")

        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ""}):
            with self.assertRaisesRegex(
                RuntimeError, "configured before Torch import"
            ):
                configure_torch_determinism(0, "cuda")

        with patch.dict(
            os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        ), patch.object(torch.cuda, "is_available", return_value=False) as available:
            with self.assertRaisesRegex(RuntimeError, "CUDA device is unavailable"):
                configure_torch_determinism(0, "cuda")
            available.assert_called_once_with()

        with patch.dict(
            os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        ), patch.object(
            torch.cuda, "is_available", return_value=True
        ), patch.object(
            torch, "manual_seed"
        ) as torch_seed, patch.object(
            torch.cuda, "manual_seed_all"
        ) as cuda_seed, patch(
            "llm_foundations_companion.tiny_v2.determinism._THREADS_CONFIGURED",
            True,
        ):
            configured = configure_torch_determinism(7, "cuda")
            self.assertIs(configured, torch)
            torch_seed.assert_called_once_with(7)
            cuda_seed.assert_called_once_with(7)

    def test_dispatch_failure_contract_and_allocated_tokenizer_identity(self) -> None:
        field_error = {"field": "seed", "message": "invalid"}
        failure = OperationFailure(
            "INVALID_REQUEST",
            "invalid request",
            field_errors=(field_error,),
            retryable=False,
        )
        field_error["message"] = "mutated"
        self.assertEqual(str(failure), "invalid request")
        self.assertEqual(failure.code, "INVALID_REQUEST")
        self.assertEqual(failure.message, "invalid request")
        self.assertEqual(
            failure.field_errors, ({"field": "seed", "message": "invalid"},)
        )
        self.assertFalse(failure.retryable)
        with self.assertRaisesRegex(OperationUnavailable, "not supported"):
            handler_for("unknown")

        with TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "train.jsonl"
            dataset.write_bytes(
                canonical_json(
                    {
                        "record_id": "train-000",
                        "scenario_group_id": "train-group-000",
                        "text": "a small training document",
                    }
                )
                + b"\n"
            )

            class MissingAllocationContext:
                output_allocations = {"tokenizer_id": None}

                @staticmethod
                def input(role: str) -> SimpleNamespace:
                    if role != "dataset.train":
                        raise AssertionError(role)
                    return SimpleNamespace(path=dataset)

                @staticmethod
                def stage_artifact(
                    role: str, data: bytes, *, filename: str
                ) -> dict[str, str]:
                    if role != "tokenizer_json" or filename != "tokenizer.json":
                        raise AssertionError((role, filename))
                    if not data:
                        raise AssertionError("empty tokenizer artifact")
                    return {"artifact_id": "tokenizer-artifact"}

            with self.assertRaisesRegex(
                ValueError, "no parent-allocated tokenizer identity"
            ):
                handler_for("tokenizer_train")(
                    {
                        "operation": "tokenizer_train",
                        "tokenizer_profile_id": "byte-v1",
                        "vocab_size": 257,
                        "seed": 0,
                    },
                    MissingAllocationContext(),
                )

    def test_cancellation_commits_before_and_after_a_training_update(self) -> None:
        before = _trainer(seed=41)
        before_weights = {
            name: tensor.detach().clone()
            for name, tensor in before.model.state_dict().items()
        }
        before_rng = before.batch_rng.get_state().clone()
        before_commits: list[CheckpointSnapshot] = []
        before_outcome = before.run(
            next_checkpoint_id=lambda: (
                "41000000-0000-4000-8000-000000000010"
            ),
            commit_checkpoint=before_commits.append,
            cancellation_reason=lambda: "cancel_before_step",
            evaluate_initial=False,
        )
        self.assertTrue(before_outcome.interrupted)
        self.assertEqual(before_outcome.state.completed_global_step, 0)
        self.assertEqual(before_outcome.interruption_reason, "cancel_before_step")
        self.assertEqual(len(before_commits), 1)
        self.assertEqual(before_commits[0].reason, "cancellation")
        self.assertEqual(before_commits[0].state.completed_global_step, 0)
        self.assertEqual(before_outcome.metrics, ())
        self.assertTrue(torch.equal(before_rng, before.batch_rng.get_state()))
        for name, tensor in before_weights.items():
            self.assertTrue(torch.equal(tensor, before.model.state_dict()[name]), name)

        after = _trainer(seed=43)
        after_commits: list[CheckpointSnapshot] = []
        cancellation_calls = 0

        def cancel_after_step() -> str | None:
            nonlocal cancellation_calls
            cancellation_calls += 1
            return "cancel_after_step" if cancellation_calls == 2 else None

        after_outcome = after.run(
            next_checkpoint_id=lambda: (
                "41000000-0000-4000-8000-000000000011"
            ),
            commit_checkpoint=after_commits.append,
            cancellation_reason=cancel_after_step,
            evaluate_initial=False,
        )
        self.assertTrue(after_outcome.interrupted)
        self.assertEqual(after_outcome.state.completed_global_step, 1)
        self.assertEqual(after_outcome.interruption_reason, "cancel_after_step")
        self.assertEqual(len(after_commits), 1)
        self.assertEqual(after_commits[0].reason, "cancellation")
        self.assertEqual(after_commits[0].state.completed_global_step, 1)
        self.assertEqual(len(after_outcome.metrics), 1)
        self.assertTrue(after.optimizer.state)

    def test_checkpoint_rejects_optimizer_rng_and_identity_tampering(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = create_resume_checkpoint(root)

            def optimizer_extra_name(directory: Path) -> None:
                def mutate(tensors: dict[str, torch.Tensor]) -> None:
                    sample = next(iter(tensors.values()))
                    tensors["unexpected.weight.exp_avg"] = sample.clone()

                rewrite_safetensors(directory, "optimizer.safetensors", mutate)

            def optimizer_bad_shape(directory: Path) -> None:
                def mutate(tensors: dict[str, torch.Tensor]) -> None:
                    name = next(
                        key for key, tensor in tensors.items() if tensor.ndim > 1
                    )
                    tensors[name] = tensors[name][:-1].contiguous()

                rewrite_safetensors(directory, "optimizer.safetensors", mutate)

            def optimizer_bad_dtype(directory: Path) -> None:
                def mutate(tensors: dict[str, torch.Tensor]) -> None:
                    name = next(iter(tensors))
                    tensors[name] = tensors[name].to(torch.float64)

                rewrite_safetensors(directory, "optimizer.safetensors", mutate)

            def parameter_name_mismatch(directory: Path) -> None:
                def mutate(value: dict[str, object]) -> None:
                    steps = value["parameter_steps"]
                    assert isinstance(steps, dict)
                    name = next(iter(steps))
                    steps["unexpected.weight"] = steps.pop(name)

                rewrite_json_payload(directory, "trainer_state.json", mutate)

            def rng_bad_length(directory: Path) -> None:
                def mutate(tensors: dict[str, torch.Tensor]) -> None:
                    tensors["batch_cpu"] = tensors["batch_cpu"][:-1].contiguous()

                rewrite_safetensors(directory, "rng.safetensors", mutate)

            def noncanonical_tokenizer(directory: Path) -> None:
                path = directory / "tokenizer.json"
                path.write_bytes(path.read_bytes() + b"\n")
                refresh_manifest_entry(directory, "tokenizer.json")

            def standard_alias_mismatch(directory: Path) -> None:
                path = directory / "manifest.json"
                value = json.loads(path.read_bytes())
                value["aliases"] = {"output.weight": "tokens.weight"}
                path.write_bytes(canonical_json(value))

            def trainer_architecture_mismatch(directory: Path) -> None:
                rewrite_json_payload(
                    directory,
                    "trainer_state.json",
                    lambda value: value.__setitem__(
                        "architecture_profile_id", TIED_PROFILE
                    ),
                )

            scenarios = {
                "optimizer_extra_name": (
                    optimizer_extra_name,
                    "missing or extra tensor names",
                ),
                "optimizer_bad_shape": (
                    optimizer_bad_shape,
                    "incompatible shape or dtype",
                ),
                "optimizer_bad_dtype": (
                    optimizer_bad_dtype,
                    "incompatible shape or dtype",
                ),
                "parameter_name_mismatch": (
                    parameter_name_mismatch,
                    "parameter steps do not match optimizer tensors",
                ),
                "rng_bad_length": (rng_bad_length, "invalid length"),
                "noncanonical_tokenizer": (
                    noncanonical_tokenizer,
                    "protected exact serialization",
                ),
                "standard_alias_mismatch": (
                    standard_alias_mismatch,
                    "aliases do not match the architecture",
                ),
                "trainer_architecture_mismatch": (
                    trainer_architecture_mismatch,
                    "architecture identity does not match",
                ),
            }
            for label, (mutation, message) in scenarios.items():
                with self.subTest(label=label):
                    candidate = clone_checkpoint(
                        fixture.directory, root / f"tampered-{label}"
                    )
                    mutation(candidate)
                    with self.assertRaisesRegex(
                        CheckpointIncompatible, message
                    ):
                        validate_checkpoint(candidate, for_resume=True)

    def test_checkpoint_write_rejects_inconsistent_optimizer_state_and_cleans_up(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = create_resume_checkpoint(root)
            baseline = copy.deepcopy(fixture.optimizer.state_dict())

            def first_state() -> dict[str, object]:
                return next(iter(fixture.optimizer.state.values()))

            def missing_scalar_step() -> TrainingState:
                first_state().pop("step")
                return fixture.trainer_state

            def mismatched_scalar_step() -> TrainingState:
                first_state()["step"] = torch.tensor(2.0)
                return fixture.trainer_state

            def missing_tensor_state() -> TrainingState:
                first_state().pop("exp_avg")
                return fixture.trainer_state

            def extra_parameter_name() -> TrainingState:
                return replace(
                    fixture.trainer_state,
                    parameter_steps={
                        **fixture.trainer_state.parameter_steps,
                        "unexpected.weight": 1,
                    },
                )

            scenarios = {
                "missing_step": (missing_scalar_step, "scalar step is missing"),
                "mismatched_step": (
                    mismatched_scalar_step,
                    "scalar step differs",
                ),
                "missing_tensor": (
                    missing_tensor_state,
                    "tensor state is missing",
                ),
                "extra_parameter": (
                    extra_parameter_name,
                    "parameter names do not match",
                ),
            }
            for label, (mutate, message) in scenarios.items():
                with self.subTest(label=label):
                    fixture.optimizer.load_state_dict(copy.deepcopy(baseline))
                    state = mutate()
                    destination = root / f"write-{label}"
                    with self.assertRaisesRegex(ValueError, message):
                        write_checkpoint(
                            destination,
                            model=fixture.model,
                            optimizer=fixture.optimizer,
                            tokenizer=fixture.tokenizer,
                            config=fixture.config,
                            trainer_state=state,
                            batch_rng=torch.Generator(
                                device="cpu"
                            ).manual_seed(47),
                        )
                    self.assertFalse(destination.exists())

    def test_weight_tied_resume_round_trip_restores_all_torch_state(self) -> None:
        original_torch_rng = torch.get_rng_state().clone()
        try:
            with TemporaryDirectory() as temporary:
                root = Path(temporary)
                torch.manual_seed(53)
                tokenizer = Tokenizer()
                config = Config(tokenizer.vocab_size, 8, 16, 1, 1)
                model = build_model(config, TIED_PROFILE)
                optimizer = create_optimizer(model, 0.001)
                training_config = TrainingConfig(
                    requested_final_step=1,
                    eval_every=1,
                    batch_size=2,
                    seed=53,
                    learning_rate=0.001,
                    architecture_profile_id=TIED_PROFILE,
                    runtime_profile="wsl-cpu",
                    device="cpu",
                    dependency_lock_sha256="e" * 64,
                    dataset_bindings=(
                        DatasetBinding(
                            "train",
                            "53000000-0000-4000-8000-000000000001",
                            "3" * 64,
                        ),
                        DatasetBinding(
                            "validation",
                            "53000000-0000-4000-8000-000000000002",
                            "4" * 64,
                        ),
                    ),
                )
                state = TrainingState.fresh(
                    training_config,
                    tokenizer_sha256=tokenizer.fingerprint(),
                    parameter_names=dict(
                        model.named_parameters(remove_duplicate=True)
                    ),
                )
                batch_rng = torch.Generator(device="cpu").manual_seed(53)
                batch_rng_state = batch_rng.get_state().clone()
                torch_rng_state = torch.get_rng_state().clone()
                model_state = {
                    name: tensor.detach().clone()
                    for name, tensor in model.state_dict().items()
                }
                write_checkpoint(
                    root / "tied-resume",
                    model=model,
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                    config=config,
                    trainer_state=state,
                    batch_rng=batch_rng,
                )
                torch.manual_seed(999)

                loaded = load_checkpoint(
                    root / "tied-resume", device="cpu", for_resume=True
                )
                self.assertEqual(loaded.architecture_profile_id, TIED_PROFILE)
                self.assertIs(
                    loaded.model.output.weight, loaded.model.tokens.weight
                )
                self.assertEqual(
                    loaded.model.output.weight.data_ptr(),
                    loaded.model.tokens.weight.data_ptr(),
                )
                for name, tensor in model_state.items():
                    self.assertTrue(
                        torch.equal(tensor, loaded.model.state_dict()[name]), name
                    )
                self.assertIsNotNone(loaded.optimizer)
                self.assertIsNotNone(loaded.batch_rng)
                self.assertTrue(
                    torch.equal(batch_rng_state, loaded.batch_rng.get_state())
                )
                self.assertTrue(torch.equal(torch_rng_state, torch.get_rng_state()))
                self.assertEqual(loaded.trainer_state, state)
                self.assertTrue(loaded.optimizer.state)
                for optimizer_state in loaded.optimizer.state.values():
                    self.assertEqual(float(optimizer_state["step"].item()), 0.0)
                    self.assertTrue(
                        torch.equal(
                            optimizer_state["exp_avg"],
                            torch.zeros_like(optimizer_state["exp_avg"]),
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            optimizer_state["exp_avg_sq"],
                            torch.zeros_like(optimizer_state["exp_avg_sq"]),
                        )
                    )
        finally:
            torch.set_rng_state(original_torch_rng)

    def test_two_subject_evaluation_and_failure_translation(self) -> None:
        from llm_foundations_companion.tiny_v2.evaluation import (
            EvaluationCancelled,
        )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = create_resume_checkpoint(root)
            dataset = root / "validation.jsonl"
            dataset.write_bytes(
                canonical_json(
                    {
                        "record_id": "evaluation-000",
                        "scenario_group_id": "evaluation-group-000",
                        "text": "hello world",
                    }
                )
                + b"\n"
            )
            descriptor = _checkpoint_descriptor(fixture)
            inputs = {
                "dataset.validation": SimpleNamespace(
                    path=dataset, artifact_id="dataset-validation"
                ),
                **_checkpoint_inputs(fixture, "subject.0.checkpoint."),
                **_checkpoint_inputs(fixture, "subject.1.checkpoint."),
            }
            resolved = {
                "dataset_manifest_sha256": hashlib.sha256(
                    dataset.read_bytes()
                ).hexdigest(),
                "subjects": [
                    {"checkpoint": descriptor},
                    {"checkpoint": descriptor},
                ],
            }
            request = {
                "operation": "evaluate",
                "subjects": [
                    {
                        "kind": "tiny_checkpoint",
                        "checkpoint_id": descriptor["checkpoint_id"],
                    },
                    {
                        "kind": "tiny_checkpoint",
                        "checkpoint_id": descriptor["checkpoint_id"],
                    },
                ],
                "dataset_id": "62000000-0000-4000-8000-000000000003",
                "split": "validation",
                "evaluation_profile_id": "tiny-nll-per-byte-v1",
            }
            context = _OperationContext(
                inputs=inputs,
                resolved=resolved,
                run_id="62000000-0000-4000-8000-000000000004",
            )
            result = handler_for("evaluate")(request, context)
            self.assertEqual(result["operation"], "evaluate")
            self.assertEqual(len(result["artifact_ids"]), 3)
            self.assertIsNotNone(result["paired_artifact_id"])
            pair = json.loads(context.artifacts["evaluation_paired"])
            self.assertEqual(pair["ordered_record_ids"], ["evaluation-000"])
            self.assertEqual(pair["delta_nll_per_utf8_byte"], 0.0)
            self.assertEqual(len(pair["subjects"]), 2)

            cancelled = _OperationContext(
                inputs=inputs,
                resolved=resolved,
                run_id="62000000-0000-4000-8000-000000000005",
            )
            cancelled.cancellation = SimpleNamespace(
                cancelled=True, reason="user_cancelled"
            )
            with patch(
                "llm_foundations_companion.tiny_v2.evaluation.evaluate_nll_per_byte",
                side_effect=EvaluationCancelled(),
            ), self.assertRaisesRegex(_Interrupted, "user_cancelled"):
                handler_for("evaluate")(request, cancelled)

            out_of_memory = _OperationContext(
                inputs=inputs,
                resolved=resolved,
                run_id="62000000-0000-4000-8000-000000000006",
            )
            with patch(
                "llm_foundations_companion.tiny_v2.evaluation.evaluate_nll_per_byte",
                side_effect=torch.cuda.OutOfMemoryError(),
            ), self.assertRaises(OperationFailure) as caught:
                handler_for("evaluate")(request, out_of_memory)
            self.assertEqual(caught.exception.code, "OUT_OF_MEMORY")

    def test_preview_generation_round_trip_and_stale_preview_rejection(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = create_resume_checkpoint(root)
            descriptor = _checkpoint_descriptor(fixture)
            checkpoint_inputs = _checkpoint_inputs(fixture, "checkpoint.")
            preview_context = _OperationContext(
                inputs=checkpoint_inputs,
                resolved={"checkpoint": descriptor},
                run_id=None,
            )
            preview_request = {
                "operation": "context_preview",
                "backend": "tiny",
                "checkpoint_id": descriptor["checkpoint_id"],
                "prompt": "Hello\nworld",
                "max_new_tokens": 2,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 17,
            }
            preview = handler_for("context_preview")(
                preview_request, preview_context
            )
            preview_path = root / "context-preview.json"
            preview_path.write_bytes(preview_context.artifacts["context_preview"])
            preview_id = str(preview["preview_artifact_id"])
            generation_inputs = {
                **checkpoint_inputs,
                "context_preview": SimpleNamespace(
                    path=preview_path,
                    artifact_id=preview_id,
                ),
            }
            generation_context = _OperationContext(
                inputs=generation_inputs,
                resolved={"checkpoint": descriptor},
                run_id="63000000-0000-4000-8000-000000000001",
            )
            generation_request = {
                key: value
                for key, value in preview_request.items()
                if key != "backend"
            }
            generation_request["operation"] = "generate"
            generation_request["preview_artifact_id"] = preview_id
            generation_request["context_preview_digest"] = preview[
                "context_preview_digest"
            ]
            generated = handler_for("generate")(
                generation_request, generation_context
            )
            self.assertEqual(generated["operation"], "generate")
            self.assertEqual(generated["checkpoint_id"], descriptor["checkpoint_id"])
            self.assertEqual(len(generated["artifact_ids"]), 1)
            artifact = json.loads(generation_context.artifacts["generation"])
            self.assertEqual(artifact["text"], generated["generated_text"])
            self.assertEqual(
                artifact["generated_token_count"],
                generated["generated_token_count"],
            )

            stale_inputs = {
                **checkpoint_inputs,
                "context_preview": SimpleNamespace(
                    path=preview_path,
                    artifact_id="63000000-0000-4000-8000-000000000099",
                ),
            }
            stale_context = _OperationContext(
                inputs=stale_inputs,
                resolved={"checkpoint": descriptor},
                run_id="63000000-0000-4000-8000-000000000002",
            )
            with self.assertRaises(OperationFailure) as caught:
                handler_for("generate")(generation_request, stale_context)
            self.assertEqual(caught.exception.code, "CONTEXT_PREVIEW_STALE")

            preview_path.write_bytes(b"not-json")
            invalid_context = _OperationContext(
                inputs={
                    **checkpoint_inputs,
                    "context_preview": SimpleNamespace(
                        path=preview_path,
                        artifact_id=preview_id,
                    ),
                },
                resolved={"checkpoint": descriptor},
                run_id="63000000-0000-4000-8000-000000000003",
            )
            with self.assertRaises(OperationFailure) as caught:
                handler_for("generate")(generation_request, invalid_context)
            self.assertEqual(caught.exception.code, "CONTEXT_PREVIEW_STALE")


if __name__ == "__main__":
    unittest.main()
