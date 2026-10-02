"""Native Tiny-v2 checkpoint fixtures shared by frozen development cases."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Callable

import torch
from safetensors.torch import load_file, save_file

from llm_foundations_companion.tiny_v2.checkpoint import (
    CheckpointPayload,
    write_checkpoint,
)
from llm_foundations_companion.tiny_v2.model import (
    Config,
    STANDARD_PROFILE,
    build_model,
)
from llm_foundations_companion.tiny_v2.tokenizer import Tokenizer
from llm_foundations_companion.tiny_v2.training import (
    DatasetBinding,
    TinyTrainer,
    TrainingConfig,
    TrainingState,
    create_optimizer,
)


@dataclass(frozen=True)
class ResumeFixture:
    directory: Path
    model: object
    optimizer: object
    tokenizer: Tokenizer
    config: Config
    trainer_state: TrainingState
    batch_rng_state: torch.Tensor
    payload: CheckpointPayload


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def create_resume_checkpoint(root: Path) -> ResumeFixture:
    torch.manual_seed(17)
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
    bindings = (
        DatasetBinding(
            "train",
            "11111111-1111-4111-8111-111111111111",
            "1" * 64,
        ),
        DatasetBinding(
            "validation",
            "22222222-2222-4222-8222-222222222222",
            "2" * 64,
        ),
    )
    training_config = TrainingConfig(
        requested_final_step=2,
        eval_every=1,
        batch_size=2,
        seed=17,
        learning_rate=0.001,
        architecture_profile_id=STANDARD_PROFILE,
        runtime_profile="wsl-cpu",
        device="cpu",
        dependency_lock_sha256="3" * 64,
        dataset_bindings=bindings,
    )
    state = TrainingState.fresh(
        training_config,
        tokenizer_sha256=tokenizer.fingerprint(),
        parameter_names=dict(model.named_parameters(remove_duplicate=True)),
    )
    batch_rng = torch.Generator(device="cpu").manual_seed(17)
    data = torch.arange(96, dtype=torch.long) % tokenizer.vocab_size
    trainer = TinyTrainer(
        model=model,
        optimizer=optimizer,
        train_data=data,
        validation_data=data.flip(0),
        batch_rng=batch_rng,
        state=state,
        device="cpu",
    )
    initial = trainer.evaluate()
    trainer.accept_checkpoint(
        trainer.checkpoint_snapshot(
            "33333333-3333-4333-8333-333333333333",
            reason="initial",
            evaluation=initial,
        )
    )
    trainer.step_once()
    directory = root / "resume"
    payload = write_checkpoint(
        directory,
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        config=config,
        trainer_state=trainer.state,
        batch_rng=batch_rng,
    )
    return ResumeFixture(
        directory=directory,
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        config=config,
        trainer_state=trainer.state,
        batch_rng_state=batch_rng.get_state().clone(),
        payload=payload,
    )


def clone_checkpoint(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def refresh_manifest_entry(directory: Path, name: str) -> None:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    raw = (directory / name).read_bytes()
    matches = [item for item in manifest["files"] if item["name"] == name]
    if len(matches) != 1:
        raise AssertionError(f"Manifest entry missing or ambiguous for {name}")
    matches[0]["size_bytes"] = len(raw)
    matches[0]["sha256"] = sha256(raw)
    manifest_path.write_bytes(canonical_json(manifest))


def rewrite_json_payload(
    directory: Path,
    name: str,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    path = directory / name
    value = json.loads(path.read_bytes())
    mutate(value)
    path.write_bytes(canonical_json(value))
    refresh_manifest_entry(directory, name)


def rewrite_safetensors(
    directory: Path,
    name: str,
    mutate: Callable[[dict[str, torch.Tensor]], None],
) -> None:
    path = directory / name
    tensors = dict(load_file(path, device="cpu"))
    mutate(tensors)
    save_file(tensors, path)
    refresh_manifest_entry(directory, name)


def make_inference_checkpoint(source: Path, destination: Path) -> str:
    destination.mkdir()
    source_manifest = json.loads((source / "manifest.json").read_bytes())
    names = ("model.safetensors", "tokenizer.json", "config.json")
    entries = []
    for name in names:
        shutil.copyfile(source / name, destination / name)
        raw = (destination / name).read_bytes()
        entries.append(
            {"name": name, "size_bytes": len(raw), "sha256": sha256(raw)}
        )
    manifest = {
        "format": "tiny-v2-checkpoint-v1",
        "portability": "inference_only",
        "files": entries,
        "aliases": source_manifest["aliases"],
    }
    raw = canonical_json(manifest)
    (destination / "manifest.json").write_bytes(raw)
    return sha256(raw)


__all__ = [
    "ResumeFixture",
    "canonical_json",
    "clone_checkpoint",
    "create_resume_checkpoint",
    "make_inference_checkpoint",
    "refresh_manifest_entry",
    "rewrite_json_payload",
    "rewrite_safetensors",
    "sha256",
]
