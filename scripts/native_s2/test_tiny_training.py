"""Frozen native cases for deterministic Tiny-v2 training boundaries."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from llm_foundations_companion.operations.tiny_training import _make_callbacks
from llm_foundations_companion.tiny_v2.model import (
    Config,
    STANDARD_PROFILE,
    build_model,
)
from llm_foundations_companion.tiny_v2.tokenizer import Tokenizer
from llm_foundations_companion.tiny_v2.training import (
    CheckpointSnapshot,
    DatasetBinding,
    Evaluation,
    TinyTrainer,
    TrainingConfig,
    TrainingState,
    create_optimizer,
)


class TinyTrainingNativeCases(unittest.TestCase):
    def test_phase_events_stay_bounded_and_reserve_terminal_boundaries(
        self,
    ) -> None:
        class Cancellation:
            reason = "user_cancelled"

            @staticmethod
            def wait(_timeout: float) -> bool:
                return True

        class Context:
            staging_path = Path(".")
            cancellation = Cancellation()

            def __init__(self) -> None:
                self.events = [("phase_changed", {"phase": "loading"})]

            def emit(self, event_type: str, payload: dict[str, object]) -> None:
                self.events.append((event_type, payload))

            @staticmethod
            def commit_checkpoint(_proposal: object) -> dict[str, object]:
                return {
                    "checkpoint_id": "00000000-0000-4000-8000-000000000032",
                    "step": 2_000,
                }

        context = Context()
        payload = SimpleNamespace(proposal_dict=lambda: {})
        with patch(
            "llm_foundations_companion.tiny_v2.checkpoint.write_checkpoint",
            return_value=payload,
        ):
            callbacks = _make_callbacks(
                context=context,
                model=object(),
                optimizer=object(),
                tokenizer=object(),
                config=object(),
                total_steps=2_000,
            )
        callbacks["bind"](SimpleNamespace(batch_rng=object()))

        for step in range(1, 2_001):
            callbacks["before_step"](step)
            callbacks["before_evaluation"](step)

        self.assertEqual(len(context.events), 30)
        callbacks["checkpoint"](
            SimpleNamespace(
                checkpoint_id="00000000-0000-4000-8000-000000000032",
                reason="final",
                state=SimpleNamespace(completed_global_step=2_000),
            )
        )
        self.assertEqual(len(context.events), 31)
        self.assertEqual(context.events[-1][1], {"phase": "checkpointing"})
        self.assertEqual(callbacks["enter_hold"](), "user_cancelled")
        self.assertEqual(len(context.events), 32)
        self.assertEqual(context.events[-1][1], {"phase": "cancellable_hold"})

    def test_s2_native_022_step_zero_evaluation_has_no_update_effect(
        self,
    ) -> None:
        torch.manual_seed(29)
        tokenizer = Tokenizer()
        config = Config(
            vocab_size=tokenizer.vocab_size,
            context=8,
            width=16,
            heads=1,
            layers=1,
        )
        model = build_model(config, STANDARD_PROFILE)
        optimizer = create_optimizer(model, 0.001)
        training_config = TrainingConfig(
            requested_final_step=1,
            eval_every=1,
            batch_size=3,
            seed=29,
            learning_rate=0.001,
            architecture_profile_id=STANDARD_PROFILE,
            runtime_profile="wsl-cpu",
            device="cpu",
            dependency_lock_sha256="4" * 64,
            dataset_bindings=(
                DatasetBinding("train", "train", "5" * 64),
                DatasetBinding("validation", "validation", "6" * 64),
            ),
        )
        state = TrainingState.fresh(
            training_config,
            tokenizer_sha256=tokenizer.fingerprint(),
            parameter_names=dict(
                model.named_parameters(remove_duplicate=True)
            ),
        )
        batch_rng = torch.Generator(device="cpu").manual_seed(29)
        data = torch.arange(128, dtype=torch.long) % tokenizer.vocab_size
        trainer = TinyTrainer(
            model=model,
            optimizer=optimizer,
            train_data=data,
            validation_data=data.flip(0),
            batch_rng=batch_rng,
            state=state,
            device="cpu",
        )

        weights_before = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        batch_rng_before = batch_rng.get_state().clone()
        torch_rng_before = torch.get_rng_state().clone()
        optimizer_state_before = len(optimizer.state)
        training_mode_before = model.training
        checkpoints: list[CheckpointSnapshot] = []

        def commit(snapshot: CheckpointSnapshot) -> None:
            checkpoints.append(snapshot)

        outcome = trainer.run(
            next_checkpoint_id=lambda: (
                "00000000-0000-4000-8000-000000000022"
            ),
            commit_checkpoint=commit,
            cancellation_reason=lambda: "test_stop_after_initial",
            evaluate_initial=True,
            checkpoint_initial=True,
        )

        self.assertTrue(outcome.interrupted)
        self.assertEqual(
            outcome.interruption_reason, "test_stop_after_initial"
        )
        self.assertEqual(outcome.state.completed_global_step, 0)
        self.assertTrue(
            all(step == 0 for step in outcome.state.parameter_steps.values())
        )
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0].reason, "initial")
        self.assertEqual(checkpoints[0].state.completed_global_step, 0)
        self.assertEqual(len(outcome.metrics), 1)
        self.assertIsInstance(outcome.metrics[0], Evaluation)
        self.assertEqual(outcome.metrics[0].step, 0)
        self.assertTrue(
            torch.isfinite(
                torch.tensor(
                    [
                        outcome.metrics[0].train_nll_token,
                        outcome.metrics[0].validation_nll_token,
                    ]
                )
            ).all().item()
        )

        weights_after = model.state_dict()
        self.assertEqual(set(weights_before), set(weights_after))
        for name, value in weights_before.items():
            self.assertTrue(
                torch.equal(value, weights_after[name]),
                name,
            )
        self.assertTrue(
            torch.equal(batch_rng_before, batch_rng.get_state())
        )
        self.assertTrue(
            torch.equal(torch_rng_before, torch.get_rng_state())
        )
        self.assertEqual(len(optimizer.state), optimizer_state_before)
        self.assertEqual(model.training, training_mode_before)

        self.s2_observation = {
            "completed_global_step": outcome.state.completed_global_step,
            "checkpoint_count": len(checkpoints),
            "checkpoint_reason": checkpoints[0].reason,
            "metric_type": type(outcome.metrics[0]).__name__,
            "weights_unchanged": True,
            "batch_rng_unchanged": True,
            "torch_rng_unchanged": True,
            "optimizer_state_entries": len(optimizer.state),
            "training_mode_restored": True,
        }


    def test_resume_cancellation_reuses_parent_boundary(self) -> None:
        torch.manual_seed(31)
        tokenizer = Tokenizer()
        config = Config(
            vocab_size=tokenizer.vocab_size,
            context=8,
            width=16,
            heads=1,
            layers=1,
        )
        model = build_model(config, STANDARD_PROFILE)
        optimizer = create_optimizer(model, 0.001)
        training_config = TrainingConfig(
            requested_final_step=2,
            eval_every=1,
            batch_size=2,
            seed=31,
            learning_rate=0.001,
            architecture_profile_id=STANDARD_PROFILE,
            runtime_profile="wsl-cpu",
            device="cpu",
            dependency_lock_sha256="7" * 64,
            dataset_bindings=(
                DatasetBinding(
                    "train",
                    "77777777-7777-4777-8777-777777777777",
                    "8" * 64,
                ),
                DatasetBinding(
                    "validation",
                    "88888888-8888-4888-8888-888888888888",
                    "9" * 64,
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
        batch_rng = torch.Generator(device="cpu").manual_seed(31)
        data = torch.arange(128, dtype=torch.long) % tokenizer.vocab_size
        source = TinyTrainer(
            model=model,
            optimizer=optimizer,
            train_data=data,
            validation_data=data.flip(0),
            batch_rng=batch_rng,
            state=state,
            device="cpu",
        )
        initial = source.evaluate()
        parent_id = "99999999-9999-4999-8999-999999999999"
        source.accept_checkpoint(
            source.checkpoint_snapshot(
                parent_id,
                reason="initial",
                evaluation=initial,
            )
        )
        source.step_once()
        parent_step = source.state.completed_global_step
        resumed_rng_before = batch_rng.get_state().clone()
        resumed_weights_before = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        resumed = TinyTrainer(
            model=model,
            optimizer=optimizer,
            train_data=data,
            validation_data=data.flip(0),
            batch_rng=batch_rng,
            state=source.state,
            device="cpu",
            last_checkpoint_id=parent_id,
            last_checkpoint_step=parent_step,
        )
        commits: list[CheckpointSnapshot] = []
        outcome = resumed.run(
            next_checkpoint_id=lambda: self.fail(
                "same-step resume cancellation allocated a new checkpoint"
            ),
            commit_checkpoint=commits.append,
            cancellation_reason=lambda: "user_cancelled",
            evaluate_initial=True,
            checkpoint_initial=False,
        )
        self.assertTrue(outcome.interrupted)
        self.assertEqual(outcome.last_checkpoint_id, parent_id)
        self.assertEqual(outcome.state.completed_global_step, parent_step)
        self.assertEqual(commits, [])
        self.assertTrue(
            torch.equal(resumed_rng_before, batch_rng.get_state())
        )
        for name, value in resumed_weights_before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)



if __name__ == "__main__":
    unittest.main()
