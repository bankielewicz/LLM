"""Deterministic TinyLM-v2 training primitives.

Imported only inside the isolated worker: the service process must not import Torch.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping

import torch

from .data import batch


ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
ADAMW_WEIGHT_DECAY = 0.01
MAX_GRADIENT_NORM = 1.0
EVALUATION_BATCHES = 8
EVALUATION_SEED_OFFSET = 1000


class TinyTrainingError(RuntimeError):
    """Stable asynchronous training failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class DatasetBinding:
    split: str
    dataset_id: str
    sha256: str

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation"}:
            raise ValueError("Tiny training binds only train and validation splits.")
        if not self.dataset_id or len(self.sha256) != 64:
            raise ValueError("Dataset binding identity is incomplete.")

    def trainer_dict(self) -> dict[str, Any]:
        return {"split": self.split, "dataset_id": self.dataset_id, "sha256": self.sha256}


@dataclass(frozen=True)
class Incumbent:
    checkpoint_id: str | None = None
    validation_nll_token: float | None = None
    step: int | None = None

    def __post_init__(self) -> None:
        values = (self.checkpoint_id, self.validation_nll_token, self.step)
        if all(value is None for value in values):
            return
        if any(value is None for value in values):
            raise ValueError("Incumbent identity, loss, and step must be all-null or complete.")
        assert self.validation_nll_token is not None
        assert self.step is not None
        if not math.isfinite(self.validation_nll_token) or self.validation_nll_token < 0:
            raise ValueError("Incumbent validation loss must be finite and nonnegative.")
        if self.step < 0:
            raise ValueError("Incumbent step must be nonnegative.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "validation_nll_token": self.validation_nll_token,
            "step": self.step,
        }


@dataclass(frozen=True)
class TrainingConfig:
    requested_final_step: int
    eval_every: int
    batch_size: int
    seed: int
    learning_rate: float
    architecture_profile_id: str
    runtime_profile: str
    device: str
    dependency_lock_sha256: str
    dataset_bindings: tuple[DatasetBinding, DatasetBinding]

    def __post_init__(self) -> None:
        if not 1 <= self.requested_final_step <= 2_147_483_647:
            raise ValueError("requested_final_step is outside the Tiny-v2 bounds.")
        if not 1 <= self.eval_every <= 500:
            raise ValueError("eval_every is outside the Tiny-v2 bounds.")
        if not 1 <= self.batch_size <= 64:
            raise ValueError("batch_size is outside the Tiny-v2 bounds.")
        if not 0 <= self.seed <= 4_294_967_295:
            raise ValueError("seed is outside the Tiny-v2 bounds.")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 0.1:
            raise ValueError("learning_rate is outside the Tiny-v2 bounds.")
        if self.architecture_profile_id not in {
            "tiny-v2-standard-v1",
            "tiny-v2-weight-tied-v1",
        }:
            raise ValueError("Unknown Tiny-v2 architecture profile.")
        if self.runtime_profile not in {"win-cpu", "win-cuda", "wsl-cpu", "wsl-cuda"}:
            raise ValueError("Unknown runtime profile.")
        expected_device = "cuda" if self.runtime_profile.endswith("cuda") else "cpu"
        if self.device != expected_device:
            raise ValueError("Runtime profile and device do not match.")
        if len(self.dependency_lock_sha256) != 64:
            raise ValueError("Dependency-lock identity is invalid.")
        if len(self.dataset_bindings) != 2:
            raise ValueError("Tiny training requires two dataset bindings.")
        if {item.split for item in self.dataset_bindings} != {"train", "validation"}:
            raise ValueError("Tiny training requires train and validation bindings.")


@dataclass(frozen=True)
class TrainingState:
    completed_global_step: int
    requested_final_step: int
    eval_every: int
    batch_size: int
    seed: int
    learning_rate: float
    parameter_steps: Mapping[str, int]
    incumbent: Incumbent
    dataset_bindings: tuple[DatasetBinding, DatasetBinding]
    tokenizer_sha256: str
    architecture_profile_id: str
    runtime_profile: str
    device: str
    dependency_lock_sha256: str

    def __post_init__(self) -> None:
        if not 0 <= self.completed_global_step <= self.requested_final_step:
            raise ValueError("Completed step exceeds requested final step.")
        if len(self.tokenizer_sha256) != 64:
            raise ValueError("Tokenizer identity is invalid.")
        if (
            not self.parameter_steps
            or any(
                step < 0 or step > 2_147_483_647
                for step in self.parameter_steps.values()
            )
        ):
            raise ValueError("Optimizer parameter steps are outside the Tiny-v2 bounds.")
        if self.completed_global_step > 2_147_483_647:
            raise ValueError("Completed step is outside the Tiny-v2 bounds.")
        if (
            self.incumbent.step is not None
            and self.incumbent.step > self.completed_global_step
        ):
            raise ValueError("Incumbent step exceeds the completed global step.")
        if self.architecture_profile_id not in {
            "tiny-v2-standard-v1",
            "tiny-v2-weight-tied-v1",
        }:
            raise ValueError("Unknown Tiny-v2 architecture profile.")
        if self.runtime_profile not in {
            "win-cpu",
            "win-cuda",
            "wsl-cpu",
            "wsl-cuda",
        }:
            raise ValueError("Unknown runtime profile.")
        expected_device = (
            "cuda" if self.runtime_profile.endswith("cuda") else "cpu"
        )
        if self.device != expected_device:
            raise ValueError("Runtime profile and device do not match.")

    @classmethod
    def fresh(
        cls,
        config: TrainingConfig,
        *,
        tokenizer_sha256: str,
        parameter_names: Iterable[str],
    ) -> "TrainingState":
        return cls(
            completed_global_step=0,
            requested_final_step=config.requested_final_step,
            eval_every=config.eval_every,
            batch_size=config.batch_size,
            seed=config.seed,
            learning_rate=config.learning_rate,
            parameter_steps={name: 0 for name in parameter_names},
            incumbent=Incumbent(),
            dataset_bindings=config.dataset_bindings,
            tokenizer_sha256=tokenizer_sha256,
            architecture_profile_id=config.architecture_profile_id,
            runtime_profile=config.runtime_profile,
            device=config.device,
            dependency_lock_sha256=config.dependency_lock_sha256,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": "tiny-v2-trainer-state-v1",
            "completed_global_step": self.completed_global_step,
            "requested_final_step": self.requested_final_step,
            "eval_every": self.eval_every,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "optimizer": {
                "name": "AdamW",
                "betas": list(ADAMW_BETAS),
                "eps": ADAMW_EPS,
                "weight_decay": ADAMW_WEIGHT_DECAY,
            },
            "parameter_steps": dict(sorted(self.parameter_steps.items())),
            "incumbent": self.incumbent.as_dict(),
            "dataset_bindings": [
                item.trainer_dict()
                for item in sorted(self.dataset_bindings, key=lambda value: value.split)
            ],
            "tokenizer_sha256": self.tokenizer_sha256,
            "architecture_profile_id": self.architecture_profile_id,
            "runtime_profile": self.runtime_profile,
            "device": self.device,
            "dependency_lock_sha256": self.dependency_lock_sha256,
        }


@dataclass(frozen=True)
class Evaluation:
    step: int
    train_nll_token: float
    validation_nll_token: float

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("Evaluation step must be nonnegative.")
        if not math.isfinite(self.train_nll_token) or not math.isfinite(
            self.validation_nll_token
        ):
            raise TinyTrainingError("NONFINITE_TRAINING_VALUE", "Evaluation loss is non-finite.")


@dataclass(frozen=True)
class StepMetric:
    step: int
    train_nll_token: float
    gradient_l2_norm: float
    elapsed_seconds: float


@dataclass(frozen=True)
class CheckpointSnapshot:
    checkpoint_id: str
    reason: str
    state: TrainingState
    evaluation: Evaluation | None


@dataclass(frozen=True)
class TrainingOutcome:
    state: TrainingState
    last_checkpoint_id: str
    metrics: tuple[StepMetric | Evaluation, ...]
    interrupted: bool
    interruption_reason: str | None


def create_optimizer(model: Any, learning_rate: float) -> Any:
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
        weight_decay=ADAMW_WEIGHT_DECAY,
    )


@torch.no_grad()
def evaluate_fixed(
    model: Any,
    data: Any,
    *,
    batch_size: int,
    seed: int,
    device: str,
    batches: int = EVALUATION_BATCHES,
) -> float:
    """Evaluate with a fresh CPU generator, leaving batch training RNG untouched."""

    was_training = model.training
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed + EVALUATION_SEED_OFFSET)
    losses: list[float] = []
    try:
        for _ in range(batches):
            inputs, targets = batch(
                data, model.cfg.context, batch_size, generator, device
            )
            _, loss = model(inputs, targets)
            value = float(loss.item())
            if not math.isfinite(value):
                raise TinyTrainingError(
                    "NONFINITE_TRAINING_VALUE", "Evaluation loss is non-finite."
                )
            losses.append(value)
    finally:
        model.train(was_training)
    return sum(losses) / len(losses)


class TinyTrainer:
    """Own model/optimizer/RNG state and expose explicit durable boundaries."""

    def __init__(
        self,
        *,
        model: Any,
        optimizer: Any,
        train_data: Any,
        validation_data: Any,
        batch_rng: Any,
        state: TrainingState,
        device: str,
        last_checkpoint_id: str | None = None,
        last_checkpoint_step: int | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_data = train_data
        self.validation_data = validation_data
        self.batch_rng = batch_rng
        self.state = state
        self.device = device
        if (last_checkpoint_id is None) != (last_checkpoint_step is None):
            raise ValueError(
                "Inherited checkpoint identity and step must be both null or complete."
            )
        if (
            last_checkpoint_step is not None
            and last_checkpoint_step != state.completed_global_step
        ):
            raise ValueError(
                "Inherited checkpoint step must equal the restored completed step."
            )
        self._started = time.monotonic()
        self._metrics: list[StepMetric | Evaluation] = []
        self._last_checkpoint_id = last_checkpoint_id
        self._last_checkpoint_step = last_checkpoint_step

    @property
    def metrics(self) -> tuple[StepMetric | Evaluation, ...]:
        return tuple(self._metrics)

    def evaluate(self) -> Evaluation:
        result = Evaluation(
            step=self.state.completed_global_step,
            train_nll_token=evaluate_fixed(
                self.model,
                self.train_data,
                batch_size=self.state.batch_size,
                seed=self.state.seed,
                device=self.device,
            ),
            validation_nll_token=evaluate_fixed(
                self.model,
                self.validation_data,
                batch_size=self.state.batch_size,
                seed=self.state.seed,
                device=self.device,
            ),
        )
        self._metrics.append(result)
        return result

    def checkpoint_snapshot(
        self,
        checkpoint_id: str,
        *,
        reason: str,
        evaluation: Evaluation | None,
    ) -> CheckpointSnapshot:
        incumbent = self.state.incumbent
        if evaluation is not None and (
            incumbent.validation_nll_token is None
            or evaluation.validation_nll_token < incumbent.validation_nll_token
        ):
            incumbent = Incumbent(
                checkpoint_id=checkpoint_id,
                validation_nll_token=evaluation.validation_nll_token,
                step=evaluation.step,
            )
        return CheckpointSnapshot(
            checkpoint_id=checkpoint_id,
            reason=reason,
            state=replace(self.state, incumbent=incumbent),
            evaluation=evaluation,
        )

    def accept_checkpoint(self, snapshot: CheckpointSnapshot) -> None:
        if snapshot.state.completed_global_step != self.state.completed_global_step:
            raise RuntimeError("Checkpoint acknowledgment arrived for stale trainer state.")
        self.state = snapshot.state
        self._last_checkpoint_id = snapshot.checkpoint_id
        self._last_checkpoint_step = snapshot.state.completed_global_step

    def step_once(self) -> StepMetric:
        self.model.train()
        inputs, targets = batch(
            self.train_data,
            self.model.cfg.context,
            self.state.batch_size,
            self.batch_rng,
            self.device,
        )
        _, loss = self.model(inputs, targets)
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise TinyTrainingError(
                "NONFINITE_TRAINING_VALUE", "Training loss is non-finite."
            )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        try:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), MAX_GRADIENT_NORM, error_if_nonfinite=True
            )
        except RuntimeError as exc:
            raise TinyTrainingError(
                "NONFINITE_TRAINING_VALUE", "Training gradient is non-finite."
            ) from exc
        gradient_value = float(gradient_norm.item())
        if not math.isfinite(gradient_value):
            raise TinyTrainingError(
                "NONFINITE_TRAINING_VALUE", "Training gradient is non-finite."
            )
        self.optimizer.step()
        completed = self.state.completed_global_step + 1
        self.state = replace(
            self.state,
            completed_global_step=completed,
            parameter_steps={name: completed for name in self.state.parameter_steps},
        )
        metric = StepMetric(
            step=completed,
            train_nll_token=loss_value,
            gradient_l2_norm=gradient_value,
            elapsed_seconds=max(0.0, time.monotonic() - self._started),
        )
        self._metrics.append(metric)
        return metric

    def run(
        self,
        *,
        next_checkpoint_id: Callable[[], str],
        commit_checkpoint: Callable[[CheckpointSnapshot], None],
        cancellation_reason: Callable[[], str | None],
        after_step: Callable[[StepMetric], None] | None = None,
        before_step: Callable[[int], None] | None = None,
        after_evaluation: Callable[[Evaluation], None] | None = None,
        before_evaluation: Callable[[int], None] | None = None,
        hold_after_step: int | None = None,
        enter_hold: Callable[[], str | None] | None = None,
        evaluate_initial: bool = True,
        checkpoint_initial: bool = True,
    ) -> TrainingOutcome:
        """Run until final step or a durable cancellation boundary.

        commit_checkpoint must return only after the service commit acknowledgment.
        A proposed incumbent becomes effective only after that acknowledgment.
        """

        def durable(reason: str, evaluation: Evaluation | None) -> None:
            snapshot = self.checkpoint_snapshot(
                next_checkpoint_id(), reason=reason, evaluation=evaluation
            )
            commit_checkpoint(snapshot)
            self.accept_checkpoint(snapshot)

        if evaluate_initial:
            if before_evaluation is not None:
                before_evaluation(self.state.completed_global_step)
            initial = self.evaluate()
            if after_evaluation is not None:
                after_evaluation(initial)
            if checkpoint_initial:
                durable("initial", initial)

        while self.state.completed_global_step < self.state.requested_final_step:
            reason = cancellation_reason()
            if reason is not None:
                if self._last_checkpoint_step != self.state.completed_global_step:
                    durable("cancellation", None)
                return self._outcome(interrupted=True, reason=reason)

            if before_step is not None:
                before_step(self.state.completed_global_step + 1)
            metric = self.step_once()
            if after_step is not None:
                after_step(metric)

            reason = cancellation_reason()
            if reason is not None:
                if self._last_checkpoint_step != self.state.completed_global_step:
                    durable("cancellation", None)
                return self._outcome(interrupted=True, reason=reason)

            step = self.state.completed_global_step
            boundary = (
                step % self.state.eval_every == 0
                or step == self.state.requested_final_step
            )
            if boundary:
                if before_evaluation is not None:
                    before_evaluation(step)
                evaluation = self.evaluate()
                if after_evaluation is not None:
                    after_evaluation(evaluation)
                durable(
                    "final" if step == self.state.requested_final_step else "cadence",
                    evaluation,
                )

            if hold_after_step is not None and step == hold_after_step:
                if not boundary:
                    raise RuntimeError(
                        "A curriculum hold must coincide with a committed checkpoint."
                    )
                if enter_hold is None:
                    raise RuntimeError("A curriculum hold needs a scheduler callback.")
                hold_reason = enter_hold()
                if hold_reason is not None:
                    return self._outcome(interrupted=True, reason=hold_reason)

        return self._outcome(interrupted=False, reason=None)

    def _outcome(self, *, interrupted: bool, reason: str | None) -> TrainingOutcome:
        if self._last_checkpoint_id is None:
            raise RuntimeError(
                "Training cannot terminate before a durable checkpoint exists."
            )
        return TrainingOutcome(
            state=self.state,
            last_checkpoint_id=self._last_checkpoint_id,
            metrics=self.metrics,
            interrupted=interrupted,
            interruption_reason=reason,
        )


__all__ = [
    "CheckpointSnapshot",
    "DatasetBinding",
    "Evaluation",
    "Incumbent",
    "StepMetric",
    "TinyTrainer",
    "TinyTrainingError",
    "TrainingConfig",
    "TrainingOutcome",
    "TrainingState",
    "create_optimizer",
    "evaluate_fixed",
]
