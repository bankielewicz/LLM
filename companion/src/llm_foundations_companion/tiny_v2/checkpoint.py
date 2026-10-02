"""Safe, portable TinyLM-v2 checkpoint serialization.

All identity, file, tensor-name, shape, dtype, and alias checks complete before
model or optimizer state is materialized.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..errors import ApiError
from ..schema import validate as validate_sealed_schema
from .model import Config, build_model
from .tokenizer import Tokenizer, serialize_tokenizer
from .training import (
    ADAMW_BETAS,
    ADAMW_EPS,
    ADAMW_WEIGHT_DECAY,
    DatasetBinding,
    Incumbent,
    TrainingState,
    create_optimizer,
)


_RESUME_FILES = (
    "model.safetensors",
    "optimizer.safetensors",
    "rng.safetensors",
    "tokenizer.json",
    "config.json",
    "trainer_state.json",
)
_INFERENCE_FILES = ("model.safetensors", "tokenizer.json", "config.json")
_ALL_FILES = frozenset((*_RESUME_FILES, "manifest.json"))


class CheckpointIncompatible(ValueError):
    code = "CHECKPOINT_INCOMPATIBLE"


@dataclass(frozen=True)
class CheckpointExpectations:
    manifest_sha256: str | None = None
    tokenizer_sha256: str | None = None
    config_sha256: str | None = None
    architecture_profile_id: str | None = None
    dataset_bindings: tuple[DatasetBinding, ...] | None = None
    runtime_profile: str | None = None
    device: str | None = None
    dependency_lock_sha256: str | None = None
    completed_global_step: int | None = None


@dataclass(frozen=True)
class CheckpointFile:
    name: str
    size: int
    sha256: str

    def proposal_dict(self) -> dict[str, Any]:
        return {"name": self.name, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True)
class CheckpointPayload:
    staging_name: str
    manifest_sha256: str
    config_sha256: str
    files: tuple[CheckpointFile, ...]

    def proposal_dict(self) -> dict[str, Any]:
        return {
            "staging_name": self.staging_name,
            "manifest_sha256": self.manifest_sha256,
            "files": [item.proposal_dict() for item in self.files],
        }


@dataclass(frozen=True)
class ValidatedCheckpoint:
    files: Mapping[str, Path]
    manifest: Mapping[str, Any]
    manifest_sha256: str
    config: Config
    config_sha256: str
    architecture_profile_id: str
    tokenizer: Tokenizer
    tokenizer_sha256: str
    trainer_state: TrainingState | None
    model_tensors: Mapping[str, torch.Tensor]
    optimizer_tensors: Mapping[str, torch.Tensor] | None
    rng_tensors: Mapping[str, torch.Tensor] | None


@dataclass(frozen=True)
class LoadedTinyCheckpoint:
    model: Any
    tokenizer: Tokenizer
    config: Config
    manifest: Mapping[str, Any]
    manifest_sha256: str
    config_sha256: str
    architecture_profile_id: str
    trainer_state: TrainingState | None
    optimizer: Any | None
    batch_rng: Any | None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _strict_json(data: bytes, label: str) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise CheckpointIncompatible(f"{label} contains a duplicate JSON key.")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise CheckpointIncompatible(f"{label} contains a non-finite number.")

    try:
        parsed = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        if isinstance(exc, CheckpointIncompatible):
            raise
        raise CheckpointIncompatible(f"{label} is not strict UTF-8 JSON.") from exc
    if not isinstance(parsed, dict):
        raise CheckpointIncompatible(f"{label} must contain a JSON object.")
    return parsed


def _validate_contract_schema(
    value: Mapping[str, Any], schema_name: str, label: str
) -> None:
    try:
        validate_sealed_schema(schema_name, dict(value))
    except ApiError as exc:
        raise CheckpointIncompatible(
            f"{label} does not satisfy the sealed schema."
        ) from exc


def _digest_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _write_durable(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def model_config_dict(config: Config, architecture_profile_id: str) -> dict[str, Any]:
    values = asdict(config)
    return {
        "format": "tiny-v2-config-v1",
        "architecture_profile_id": architecture_profile_id,
        "vocab_size": values["vocab_size"],
        "context": values["context"],
        "width": values["width"],
        "heads": values["heads"],
        "layers": values["layers"],
    }


def _parse_config(value: Mapping[str, Any]) -> tuple[Config, str]:
    required = {
        "format",
        "architecture_profile_id",
        "vocab_size",
        "context",
        "width",
        "heads",
        "layers",
    }
    if set(value) != required or value.get("format") != "tiny-v2-config-v1":
        raise CheckpointIncompatible("config.json has invalid fields or format.")
    architecture = value["architecture_profile_id"]
    if architecture not in {"tiny-v2-standard-v1", "tiny-v2-weight-tied-v1"}:
        raise CheckpointIncompatible("config.json has an unknown architecture profile.")
    names = ("vocab_size", "context", "width", "heads", "layers")
    if any(isinstance(value[name], bool) or not isinstance(value[name], int) for name in names):
        raise CheckpointIncompatible("config.json model dimensions must be integers.")
    try:
        config = Config(**{name: value[name] for name in names})
    except (TypeError, ValueError) as exc:
        raise CheckpointIncompatible("config.json model dimensions are invalid.") from exc
    if not (
        257 <= config.vocab_size <= 1024
        and 8 <= config.context <= 512
        and 16 <= config.width <= 512
        and 1 <= config.heads <= 16
        and 1 <= config.layers <= 12
    ):
        raise CheckpointIncompatible("config.json model dimensions exceed sealed bounds.")
    parameters = (
        2 * config.vocab_size * config.width
        + config.context * config.width
        + config.layers * (12 * config.width * config.width + 13 * config.width)
        + 2 * config.width
    )
    if architecture == "tiny-v2-weight-tied-v1":
        parameters -= config.vocab_size * config.width
    if parameters > 10_000_000:
        raise CheckpointIncompatible("config.json exceeds the Tiny-v2 parameter cap.")
    return config, architecture


def load_config_bytes(raw: bytes) -> tuple[Config, str, str]:
    """Validate portable config bytes without loading model weights."""

    config, architecture = _parse_config(_strict_json(raw, "config.json"))
    return config, architecture, _sha256(raw)


def _parse_binding(value: object) -> DatasetBinding:
    if not isinstance(value, dict) or set(value) != {"split", "dataset_id", "sha256"}:
        raise CheckpointIncompatible("trainer_state.json has an invalid dataset binding.")
    try:
        return DatasetBinding(
            split=value["split"], dataset_id=value["dataset_id"], sha256=value["sha256"]
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointIncompatible("trainer_state.json has an invalid dataset binding.") from exc


def _parse_trainer_state(value: Mapping[str, Any]) -> TrainingState:
    required = {
        "format",
        "completed_global_step",
        "requested_final_step",
        "eval_every",
        "batch_size",
        "seed",
        "learning_rate",
        "optimizer",
        "parameter_steps",
        "incumbent",
        "dataset_bindings",
        "tokenizer_sha256",
        "architecture_profile_id",
        "runtime_profile",
        "device",
        "dependency_lock_sha256",
    }
    if set(value) != required or value.get("format") != "tiny-v2-trainer-state-v1":
        raise CheckpointIncompatible("trainer_state.json has invalid fields or format.")
    optimizer = value.get("optimizer")
    if optimizer != {
        "name": "AdamW",
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
    }:
        raise CheckpointIncompatible("trainer_state.json has incompatible optimizer settings.")
    integer_names = (
        "completed_global_step",
        "requested_final_step",
        "eval_every",
        "batch_size",
        "seed",
    )
    if any(
        isinstance(value[name], bool) or not isinstance(value[name], int)
        for name in integer_names
    ):
        raise CheckpointIncompatible("trainer_state.json integer fields are invalid.")
    learning_rate = value["learning_rate"]
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
    ):
        raise CheckpointIncompatible("trainer_state.json learning rate is invalid.")
    raw_steps = value["parameter_steps"]
    if not isinstance(raw_steps, dict) or not raw_steps:
        raise CheckpointIncompatible("trainer_state.json parameter steps are invalid.")
    parameter_steps: dict[str, int] = {}
    for name, step in raw_steps.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(step, bool)
            or not isinstance(step, int)
        ):
            raise CheckpointIncompatible("trainer_state.json parameter steps are invalid.")
        parameter_steps[name] = step
    raw_incumbent = value["incumbent"]
    if not isinstance(raw_incumbent, dict) or set(raw_incumbent) != {
        "checkpoint_id",
        "validation_nll_token",
        "step",
    }:
        raise CheckpointIncompatible("trainer_state.json incumbent is invalid.")
    try:
        incumbent = Incumbent(
            checkpoint_id=raw_incumbent["checkpoint_id"],
            validation_nll_token=raw_incumbent["validation_nll_token"],
            step=raw_incumbent["step"],
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointIncompatible("trainer_state.json incumbent is invalid.") from exc
    raw_bindings = value["dataset_bindings"]
    if not isinstance(raw_bindings, list) or len(raw_bindings) != 2:
        raise CheckpointIncompatible("trainer_state.json dataset bindings are invalid.")
    bindings = tuple(_parse_binding(item) for item in raw_bindings)
    try:
        return TrainingState(
            completed_global_step=value["completed_global_step"],
            requested_final_step=value["requested_final_step"],
            eval_every=value["eval_every"],
            batch_size=value["batch_size"],
            seed=value["seed"],
            learning_rate=float(learning_rate),
            parameter_steps=parameter_steps,
            incumbent=incumbent,
            dataset_bindings=bindings,  # type: ignore[arg-type]
            tokenizer_sha256=value["tokenizer_sha256"],
            architecture_profile_id=value["architecture_profile_id"],
            runtime_profile=value["runtime_profile"],
            device=value["device"],
            dependency_lock_sha256=value["dependency_lock_sha256"],
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointIncompatible("trainer_state.json identities are invalid.") from exc


def _unique_named_parameters(model: Any) -> dict[str, Any]:
    return dict(model.named_parameters(remove_duplicate=True))


def _model_tensor_specs(config: Config, architecture: str) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    c, v, w = config.context, config.vocab_size, config.width
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
        "tokens.weight": ((v, w), torch.float32),
        "positions.weight": ((c, w), torch.float32),
        "norm.weight": ((w,), torch.float32),
        "norm.bias": ((w,), torch.float32),
    }
    if architecture == "tiny-v2-standard-v1":
        specs["output.weight"] = ((v, w), torch.float32)
    for index in range(config.layers):
        prefix = f"blocks.{index}"
        specs.update(
            {
                f"{prefix}.norm1.weight": ((w,), torch.float32),
                f"{prefix}.norm1.bias": ((w,), torch.float32),
                f"{prefix}.attention.qkv.weight": ((3 * w, w), torch.float32),
                f"{prefix}.attention.qkv.bias": ((3 * w,), torch.float32),
                f"{prefix}.attention.output.weight": ((w, w), torch.float32),
                f"{prefix}.attention.output.bias": ((w,), torch.float32),
                f"{prefix}.attention.mask": ((c, c), torch.bool),
                f"{prefix}.norm2.weight": ((w,), torch.float32),
                f"{prefix}.norm2.bias": ((w,), torch.float32),
                f"{prefix}.mlp.0.weight": ((4 * w, w), torch.float32),
                f"{prefix}.mlp.0.bias": ((4 * w,), torch.float32),
                f"{prefix}.mlp.2.weight": ((w, 4 * w), torch.float32),
                f"{prefix}.mlp.2.bias": ((w,), torch.float32),
            }
        )
    return specs


def _validate_tensor_mapping(
    tensors: Mapping[str, torch.Tensor],
    specs: Mapping[str, tuple[tuple[int, ...], torch.dtype]],
    label: str,
) -> None:
    if set(tensors) != set(specs):
        raise CheckpointIncompatible(f"{label} has missing or extra tensor names.")
    for name, (shape, dtype) in specs.items():
        tensor = tensors[name]
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise CheckpointIncompatible(f"{label} tensor {name} has incompatible shape or dtype.")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise CheckpointIncompatible(f"{label} tensor {name} contains a non-finite value.")
        if name.endswith(".attention.mask"):
            causal = torch.tril(torch.ones(shape, dtype=torch.bool))
            if not torch.equal(tensor, causal):
                raise CheckpointIncompatible(
                    f"{label} tensor {name} does not contain the fixed causal mask."
                )


def _load_safetensors(path: Path, label: str) -> dict[str, torch.Tensor]:
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            names = list(handle.keys())
            tensors = {name: handle.get_tensor(name) for name in names}
    except Exception as exc:
        raise CheckpointIncompatible(f"{label} is not a valid safetensors file.") from exc
    return tensors


def _expected_optimizer_specs(
    config: Config, architecture: str
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    model_specs = _model_tensor_specs(config, architecture)
    model_specs = {
        name: spec
        for name, spec in model_specs.items()
        if not name.endswith(".mask") and name != "output.weight"
        or architecture == "tiny-v2-standard-v1" and name == "output.weight"
    }
    result: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
    for name, spec in model_specs.items():
        result[f"{name}.exp_avg"] = spec
        result[f"{name}.exp_avg_sq"] = spec
    return result


def _validate_manifest(value: Mapping[str, Any]) -> tuple[str, dict[str, tuple[int, str]]]:
    if set(value) != {"format", "portability", "files", "aliases"}:
        raise CheckpointIncompatible("manifest.json has invalid fields.")
    if value["format"] != "tiny-v2-checkpoint-v1":
        raise CheckpointIncompatible("manifest.json has an incompatible format.")
    portability = value["portability"]
    required_names = _RESUME_FILES if portability == "resume" else _INFERENCE_FILES
    if portability not in {"resume", "inference_only"}:
        raise CheckpointIncompatible("manifest.json has an invalid portability.")
    files = value["files"]
    if not isinstance(files, list) or len(files) != len(required_names):
        raise CheckpointIncompatible("manifest.json has an invalid file list.")
    entries: dict[str, tuple[int, str]] = {}
    for item in files:
        if not isinstance(item, dict) or set(item) != {"name", "size_bytes", "sha256"}:
            raise CheckpointIncompatible("manifest.json has an invalid file entry.")
        name, size, digest = item["name"], item["size_bytes"], item["sha256"]
        if (
            name in entries
            or name not in required_names
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 1
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise CheckpointIncompatible("manifest.json has an invalid file entry.")
        entries[name] = (size, digest)
    if set(entries) != set(required_names):
        raise CheckpointIncompatible("manifest.json has missing or extra files.")
    aliases = value["aliases"]
    expected_aliases = (
        {"output.weight": "tokens.weight"}
        if value.get("aliases") and "output.weight" in value["aliases"]
        else {}
    )
    if aliases not in ({}, {"output.weight": "tokens.weight"}) or aliases != expected_aliases:
        raise CheckpointIncompatible("manifest.json has invalid aliases.")
    return portability, entries


def _resolve_files(source: Path | Mapping[str, Path]) -> dict[str, Path]:
    if isinstance(source, Path):
        if not source.is_dir():
            raise CheckpointIncompatible("Checkpoint source is not a directory.")
        files = {path.name: path for path in source.iterdir() if path.is_file()}
    else:
        files = {str(name): Path(path) for name, path in source.items()}
    if "manifest.json" not in files:
        raise CheckpointIncompatible("Checkpoint has no manifest.json.")
    if not set(files) <= _ALL_FILES:
        raise CheckpointIncompatible("Checkpoint contains an unsafe or unknown file.")
    if any(not path.is_file() for path in files.values()):
        raise CheckpointIncompatible("Checkpoint references a missing file.")
    return files


def _check_expected(
    validated: ValidatedCheckpoint, expected: CheckpointExpectations | None
) -> None:
    if expected is None:
        return
    comparisons = (
        ("manifest digest", expected.manifest_sha256, validated.manifest_sha256),
        ("tokenizer digest", expected.tokenizer_sha256, validated.tokenizer_sha256),
        ("config digest", expected.config_sha256, validated.config_sha256),
        (
            "architecture profile",
            expected.architecture_profile_id,
            validated.architecture_profile_id,
        ),
    )
    for label, wanted, actual in comparisons:
        if wanted is not None and wanted != actual:
            raise CheckpointIncompatible(f"Checkpoint {label} does not match.")
    state = validated.trainer_state
    state_comparisons = (
        ("runtime profile", expected.runtime_profile, state.runtime_profile if state else None),
        ("device", expected.device, state.device if state else None),
        (
            "dependency lock",
            expected.dependency_lock_sha256,
            state.dependency_lock_sha256 if state else None,
        ),
        (
            "completed step",
            expected.completed_global_step,
            state.completed_global_step if state else None,
        ),
    )
    for label, wanted, actual in state_comparisons:
        if wanted is not None and wanted != actual:
            raise CheckpointIncompatible(f"Checkpoint {label} does not match.")
    if expected.dataset_bindings is not None:
        if state is None or {
            (item.split, item.dataset_id, item.sha256)
            for item in expected.dataset_bindings
        } != {
            (item.split, item.dataset_id, item.sha256)
            for item in state.dataset_bindings
        }:
            raise CheckpointIncompatible("Checkpoint dataset bindings do not match.")


def validate_checkpoint(
    source: Path | Mapping[str, Path],
    *,
    expected: CheckpointExpectations | None = None,
    for_resume: bool = False,
) -> ValidatedCheckpoint:
    files = _resolve_files(source)
    manifest_bytes = files["manifest.json"].read_bytes()
    manifest_sha256 = _sha256(manifest_bytes)
    manifest = _strict_json(manifest_bytes, "manifest.json")
    _validate_contract_schema(
        manifest, "checkpoint-manifest.schema.json", "manifest.json"
    )
    portability, entries = _validate_manifest(manifest)
    if for_resume and portability != "resume":
        raise CheckpointIncompatible("Checkpoint is not resumable.")
    if set(files) != set(entries) | {"manifest.json"}:
        raise CheckpointIncompatible("Checkpoint directory and manifest file sets differ.")
    for name, (size, digest) in entries.items():
        actual_size, actual_digest = _digest_file(files[name])
        if (actual_size, actual_digest) != (size, digest):
            raise CheckpointIncompatible(
                f"Checkpoint file {name} fails size or digest validation."
            )

    config_bytes = files["config.json"].read_bytes()
    config, architecture, config_sha256 = load_config_bytes(config_bytes)

    tokenizer_bytes = files["tokenizer.json"].read_bytes()
    tokenizer_value = _strict_json(tokenizer_bytes, "tokenizer.json")
    try:
        tokenizer = Tokenizer.from_dict(tokenizer_value)
        tokenizer_sha256 = tokenizer.fingerprint()
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointIncompatible("tokenizer.json is incompatible.") from exc
    if tokenizer_bytes != serialize_tokenizer(tokenizer):
        raise CheckpointIncompatible(
            "tokenizer.json does not use the protected exact serialization."
        )
    if tokenizer.vocab_size != config.vocab_size:
        raise CheckpointIncompatible("Tokenizer vocabulary and model config differ.")

    expected_aliases = (
        {"output.weight": "tokens.weight"}
        if architecture == "tiny-v2-weight-tied-v1"
        else {}
    )
    if manifest["aliases"] != expected_aliases:
        raise CheckpointIncompatible(
            "Checkpoint aliases do not match the architecture."
        )

    trainer_state = None
    if portability == "resume":
        trainer_value = _strict_json(
            files["trainer_state.json"].read_bytes(), "trainer_state.json"
        )
        _validate_contract_schema(
            trainer_value,
            "tiny-trainer-state.schema.json",
            "trainer_state.json",
        )
        trainer_state = _parse_trainer_state(trainer_value)
        if trainer_state.tokenizer_sha256 != tokenizer_sha256:
            raise CheckpointIncompatible(
                "Trainer-state tokenizer identity does not match."
            )
        if trainer_state.architecture_profile_id != architecture:
            raise CheckpointIncompatible(
                "Trainer-state architecture identity does not match."
            )

    # Reject metadata/identity mismatches before parsing any tensor payload.
    metadata = ValidatedCheckpoint(
        files=files,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        config=config,
        config_sha256=config_sha256,
        architecture_profile_id=architecture,
        tokenizer=tokenizer,
        tokenizer_sha256=tokenizer_sha256,
        trainer_state=trainer_state,
        model_tensors={},
        optimizer_tensors=None,
        rng_tensors=None,
    )
    _check_expected(metadata, expected)

    model_tensors = _load_safetensors(
        files["model.safetensors"], "model.safetensors"
    )
    _validate_tensor_mapping(
        model_tensors,
        _model_tensor_specs(config, architecture),
        "model.safetensors",
    )

    optimizer_tensors = None
    rng_tensors = None
    if portability == "resume":
        assert trainer_state is not None
        optimizer_tensors = _load_safetensors(
            files["optimizer.safetensors"], "optimizer.safetensors"
        )
        optimizer_specs = _expected_optimizer_specs(config, architecture)
        _validate_tensor_mapping(
            optimizer_tensors, optimizer_specs, "optimizer.safetensors"
        )
        parameter_names = {
            name[: -len(".exp_avg")]
            for name in optimizer_tensors
            if name.endswith(".exp_avg")
        }
        if set(trainer_state.parameter_steps) != parameter_names:
            raise CheckpointIncompatible(
                "Trainer-state parameter steps do not match optimizer tensors."
            )
        if any(
            step != trainer_state.completed_global_step
            for step in trainer_state.parameter_steps.values()
        ):
            raise CheckpointIncompatible(
                "Optimizer steps do not match completed global step."
            )

        rng_tensors = _load_safetensors(
            files["rng.safetensors"], "rng.safetensors"
        )
        required_rng = {"batch_cpu", "torch_cpu"}
        if trainer_state.device == "cuda":
            required_rng.add("cuda.0")
        if set(rng_tensors) != required_rng:
            raise CheckpointIncompatible(
                "rng.safetensors has invalid tensor names."
            )
        for name, tensor in rng_tensors.items():
            if tensor.dtype != torch.uint8 or tensor.ndim != 1:
                raise CheckpointIncompatible(
                    f"RNG tensor {name} has invalid shape or dtype."
                )
        try:
            cpu_probe = torch.Generator(device="cpu")
            expected_cpu_length = cpu_probe.get_state().numel()
            for name in ("batch_cpu", "torch_cpu"):
                if rng_tensors[name].numel() != expected_cpu_length:
                    raise CheckpointIncompatible(
                        f"RNG tensor {name} has an invalid length."
                    )
                cpu_probe.set_state(rng_tensors[name])
            if trainer_state.device == "cuda":
                if not torch.cuda.is_available() or torch.cuda.current_device() != 0:
                    raise CheckpointIncompatible(
                        "The selected CUDA device is not available as cuda.0."
                    )
                cuda_probe = torch.Generator(device="cuda:0")
                if rng_tensors["cuda.0"].numel() != cuda_probe.get_state().numel():
                    raise CheckpointIncompatible(
                        "RNG tensor cuda.0 has an invalid length."
                    )
                cuda_probe.set_state(rng_tensors["cuda.0"])
        except CheckpointIncompatible:
            raise
        except (RuntimeError, ValueError) as exc:
            raise CheckpointIncompatible(
                "rng.safetensors contains an invalid generator state."
            ) from exc

    return ValidatedCheckpoint(
        files=files,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        config=config,
        config_sha256=config_sha256,
        architecture_profile_id=architecture,
        tokenizer=tokenizer,
        tokenizer_sha256=tokenizer_sha256,
        trainer_state=trainer_state,
        model_tensors=model_tensors,
        optimizer_tensors=optimizer_tensors,
        rng_tensors=rng_tensors,
    )

def _portable_model_state(model: Any, architecture: str) -> dict[str, torch.Tensor]:
    state = {
        name: tensor.detach().to("cpu").contiguous()
        for name, tensor in model.state_dict().items()
    }
    if architecture == "tiny-v2-weight-tied-v1":
        state.pop("output.weight", None)
    return state


def _portable_optimizer_state(
    model: Any, optimizer: Any, trainer_state: TrainingState
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, parameter in _unique_named_parameters(model).items():
        state = optimizer.state.get(parameter, {})
        expected_step = trainer_state.parameter_steps[name]
        raw_step = state.get("step")
        if raw_step is None:
            if expected_step != 0 or state:
                raise ValueError(
                    f"Optimizer scalar step is missing for parameter {name}."
                )
        else:
            if (
                not isinstance(raw_step, torch.Tensor)
                or raw_step.numel() != 1
                or not math.isfinite(float(raw_step.item()))
                or float(raw_step.item()) != expected_step
            ):
                raise ValueError(
                    f"Optimizer scalar step differs for parameter {name}."
                )
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if value is None:
                if expected_step != 0:
                    raise ValueError(
                        f"Optimizer tensor state is missing for parameter {name}."
                    )
                value = torch.zeros_like(parameter)
            result[f"{name}.{key}"] = value.detach().to("cpu").contiguous()
    if set(trainer_state.parameter_steps) != set(_unique_named_parameters(model)):
        raise ValueError("Trainer-state parameter names do not match the model.")
    return result


def _portable_rng_state(batch_rng: Any, device: str) -> dict[str, torch.Tensor]:
    result = {
        "batch_cpu": batch_rng.get_state().to("cpu").contiguous(),
        "torch_cpu": torch.get_rng_state().to("cpu").contiguous(),
    }
    if device == "cuda":
        if torch.cuda.current_device() != 0:
            raise ValueError("The qualified worker must expose its selected CUDA device as cuda.0.")
        result["cuda.0"] = torch.cuda.get_rng_state(0).to("cpu").contiguous()
    return result


def write_checkpoint(
    directory: Path,
    *,
    model: Any,
    optimizer: Any,
    tokenizer: Tokenizer,
    config: Config,
    trainer_state: TrainingState,
    batch_rng: Any,
) -> CheckpointPayload:
    """Write one new durable checkpoint directory and return proposal metadata."""

    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Checkpoint staging directory already exists: {directory}")
    directory.mkdir(parents=False, exist_ok=False)
    architecture = trainer_state.architecture_profile_id
    try:
        model_tensors = _portable_model_state(model, architecture)
        optimizer_tensors = _portable_optimizer_state(model, optimizer, trainer_state)
        rng_tensors = _portable_rng_state(batch_rng, trainer_state.device)
        save_file(model_tensors, directory / "model.safetensors")
        save_file(optimizer_tensors, directory / "optimizer.safetensors")
        save_file(rng_tensors, directory / "rng.safetensors")
        _write_durable(directory / "tokenizer.json", serialize_tokenizer(tokenizer))
        config_bytes = _json_bytes(model_config_dict(config, architecture))
        _write_durable(directory / "config.json", config_bytes)
        _write_durable(directory / "trainer_state.json", _json_bytes(trainer_state.as_dict()))
        for name in ("model.safetensors", "optimizer.safetensors", "rng.safetensors"):
            with (directory / name).open("rb") as handle:
                os.fsync(handle.fileno())
        entries = []
        for name in _RESUME_FILES:
            size, digest = _digest_file(directory / name)
            entries.append({"name": name, "size_bytes": size, "sha256": digest})
        manifest = {
            "format": "tiny-v2-checkpoint-v1",
            "portability": "resume",
            "files": entries,
            "aliases": (
                {"output.weight": "tokens.weight"}
                if architecture == "tiny-v2-weight-tied-v1"
                else {}
            ),
        }
        manifest_bytes = _json_bytes(manifest)
        _write_durable(directory / "manifest.json", manifest_bytes)
        _fsync_directory(directory)
        validated = validate_checkpoint(directory, for_resume=True)
        if (
            validated.manifest_sha256 != _sha256(manifest_bytes)
            or validated.config_sha256 != _sha256(config_bytes)
            or validated.trainer_state != trainer_state
        ):
            raise ValueError(
                "Written checkpoint validation did not preserve its identities."
            )
        files = []
        for name in (*_RESUME_FILES, "manifest.json"):
            size, digest = _digest_file(directory / name)
            files.append(
                CheckpointFile(name=name, size=size, sha256=digest)
            )
        return CheckpointPayload(
            staging_name=directory.name,
            manifest_sha256=_sha256(manifest_bytes),
            config_sha256=_sha256(config_bytes),
            files=tuple(files),
        )
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def materialize_model(validated: ValidatedCheckpoint, *, device: str) -> Any:
    model = build_model(
        validated.config,
        architecture_profile_id=validated.architecture_profile_id,
    )
    state = dict(validated.model_tensors)
    if validated.architecture_profile_id == "tiny-v2-weight-tied-v1":
        state["output.weight"] = state["tokens.weight"]
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise CheckpointIncompatible("Model tensors could not be materialized.") from exc
    if validated.architecture_profile_id == "tiny-v2-weight-tied-v1":
        if (
            model.output.weight is not model.tokens.weight
            or model.output.weight.data_ptr() != model.tokens.weight.data_ptr()
        ):
            raise CheckpointIncompatible(
                "Tied model aliases were not preserved during materialization."
            )
    return model.to(device)


def _restore_optimizer(
    model: Any,
    tensors: Mapping[str, torch.Tensor],
    state: TrainingState,
) -> Any:
    optimizer = create_optimizer(model, state.learning_rate)
    for name, parameter in _unique_named_parameters(model).items():
        optimizer.state[parameter] = {
            "step": torch.tensor(
                float(state.parameter_steps[name]), device=parameter.device
            ),
            "exp_avg": tensors[f"{name}.exp_avg"].to(parameter.device).clone(),
            "exp_avg_sq": tensors[f"{name}.exp_avg_sq"].to(parameter.device).clone(),
        }
    return optimizer


def load_checkpoint(
    source: Path | Mapping[str, Path],
    *,
    device: str,
    expected: CheckpointExpectations | None = None,
    for_resume: bool = False,
) -> LoadedTinyCheckpoint:
    validated = validate_checkpoint(source, expected=expected, for_resume=for_resume)
    if (
        for_resume
        and validated.trainer_state is not None
        and validated.trainer_state.device != device
    ):
        raise CheckpointIncompatible(
            "Checkpoint device does not match the requested resume device."
        )
    model = materialize_model(validated, device=device)
    optimizer = None
    batch_rng = None
    if for_resume:
        assert validated.trainer_state is not None
        assert validated.optimizer_tensors is not None
        assert validated.rng_tensors is not None
        optimizer = _restore_optimizer(
            model, validated.optimizer_tensors, validated.trainer_state
        )
        batch_rng = torch.Generator(device="cpu")
        batch_rng.set_state(validated.rng_tensors["batch_cpu"])
        torch.set_rng_state(validated.rng_tensors["torch_cpu"])
        if device == "cuda":
            cuda_items = [
                (name, tensor)
                for name, tensor in validated.rng_tensors.items()
                if name.startswith("cuda.")
            ]
            name, tensor = cuda_items[0]
            index = int(name.split(".", 1)[1])
            if index != torch.cuda.current_device():
                raise CheckpointIncompatible("Checkpoint CUDA RNG device does not match.")
            torch.cuda.set_rng_state(tensor, index)
    return LoadedTinyCheckpoint(
        model=model,
        tokenizer=validated.tokenizer,
        config=validated.config,
        manifest=validated.manifest,
        manifest_sha256=validated.manifest_sha256,
        config_sha256=validated.config_sha256,
        architecture_profile_id=validated.architecture_profile_id,
        trainer_state=validated.trainer_state,
        optimizer=optimizer,
        batch_rng=batch_rng,
    )


__all__ = [
    "CheckpointExpectations",
    "CheckpointFile",
    "CheckpointIncompatible",
    "CheckpointPayload",
    "LoadedTinyCheckpoint",
    "ValidatedCheckpoint",
    "load_checkpoint",
    "load_config_bytes",
    "materialize_model",
    "model_config_dict",
    "validate_checkpoint",
    "write_checkpoint",
]
