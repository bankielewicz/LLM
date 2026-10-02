"""Frozen native cases for safe Tiny-v2 checkpoint loading."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from _tiny_fixtures import (
    canonical_json,
    clone_checkpoint,
    create_resume_checkpoint,
    make_inference_checkpoint,
    rewrite_json_payload,
    rewrite_safetensors,
    sha256,
)
from llm_foundations_companion.schema import validate as validate_schema
from llm_foundations_companion.tiny_v2.checkpoint import (
    CheckpointIncompatible,
    load_checkpoint,
    validate_checkpoint,
)


class TinyCheckpointNativeCases(unittest.TestCase):
    def test_s2_native_018_tampered_checkpoint_matrix(self) -> None:
        rejected: dict[str, str] = {}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = create_resume_checkpoint(root)
            valid = validate_checkpoint(fixture.directory, for_resume=True)
            self.assertEqual(valid.trainer_state, fixture.trainer_state)
            validate_schema(
                "checkpoint-manifest.schema.json",
                json.loads((fixture.directory / "manifest.json").read_bytes()),
            )
            validate_schema(
                "tiny-trainer-state.schema.json",
                json.loads(
                    (fixture.directory / "trainer_state.json").read_bytes()
                ),
            )
            self.assertEqual(
                {item.name for item in fixture.payload.files},
                {
                    "model.safetensors",
                    "optimizer.safetensors",
                    "rng.safetensors",
                    "tokenizer.json",
                    "config.json",
                    "trainer_state.json",
                    "manifest.json",
                },
            )

            def reject(
                label: str,
                mutation: object,
            ) -> None:
                candidate = clone_checkpoint(
                    fixture.directory, root / f"tampered-{label}"
                )
                mutation(candidate)
                with self.assertRaises(
                    CheckpointIncompatible, msg=label
                ) as raised:
                    validate_checkpoint(candidate, for_resume=True)
                rejected[label] = str(raised.exception)

            def bad_digest(directory: Path) -> None:
                path = directory / "model.safetensors"
                path.write_bytes(path.read_bytes() + b"\x00")

            def bad_name(directory: Path) -> None:
                def mutate(tensors: dict[str, torch.Tensor]) -> None:
                    tensors["unexpected.bias"] = tensors.pop("norm.bias")

                rewrite_safetensors(
                    directory, "model.safetensors", mutate
                )

            def bad_shape(directory: Path) -> None:
                def mutate(tensors: dict[str, torch.Tensor]) -> None:
                    tensors["positions.weight"] = tensors[
                        "positions.weight"
                    ][:-1].contiguous()

                rewrite_safetensors(
                    directory, "model.safetensors", mutate
                )

            def bad_dtype(directory: Path) -> None:
                def mutate(tensors: dict[str, torch.Tensor]) -> None:
                    tensors["norm.bias"] = tensors["norm.bias"].to(
                        torch.float64
                    )

                rewrite_safetensors(
                    directory, "model.safetensors", mutate
                )

            def bad_causal_mask(directory: Path) -> None:
                def mutate(tensors: dict[str, torch.Tensor]) -> None:
                    mask = tensors[
                        "blocks.0.attention.mask"
                    ].clone()
                    mask[0, 1] = True
                    tensors["blocks.0.attention.mask"] = mask

                rewrite_safetensors(
                    directory, "model.safetensors", mutate
                )

            def missing_optimizer(directory: Path) -> None:
                (directory / "optimizer.safetensors").unlink()

            def alias_cycle(directory: Path) -> None:
                path = directory / "manifest.json"
                manifest = json.loads(path.read_bytes())
                manifest["aliases"] = {
                    "output.weight": "tokens.weight",
                    "tokens.weight": "output.weight",
                }
                path.write_bytes(canonical_json(manifest))

            def internal_identity_mismatch(directory: Path) -> None:
                def mutate(value: dict[str, object]) -> None:
                    value["tokenizer_sha256"] = "f" * 64

                rewrite_json_payload(
                    directory, "trainer_state.json", mutate
                )

            def invalid_dataset_identity(directory: Path) -> None:
                def mutate(value: dict[str, object]) -> None:
                    bindings = value["dataset_bindings"]
                    assert isinstance(bindings, list)
                    bindings[0]["dataset_id"] = "not-a-uuid"

                rewrite_json_payload(
                    directory, "trainer_state.json", mutate
                )

            scenarios = {
                "bad_digest": bad_digest,
                "bad_tensor_name": bad_name,
                "bad_tensor_shape": bad_shape,
                "bad_tensor_dtype": bad_dtype,
                "digest_valid_bad_causal_mask": bad_causal_mask,
                "missing_optimizer": missing_optimizer,
                "alias_cycle": alias_cycle,
                "digest_valid_internal_identity": internal_identity_mismatch,
                "schema_invalid_dataset_identity": invalid_dataset_identity,
            }
            for label, mutation in scenarios.items():
                with self.subTest(label=label):
                    reject(label, mutation)

        self.assertEqual(set(rejected), set(scenarios))
        self.assertTrue(
            all(message for message in rejected.values()),
            rejected,
        )
        self.s2_observation = {
            "valid_file_count": 7,
            "rejected_tamper_classes": rejected,
            "digest_valid_semantic_tamper": [
                "digest_valid_bad_causal_mask",
                "digest_valid_internal_identity",
            ],
            "effective_tensor_use_after_rejection": False,
        }

    def test_s2_native_019_inference_only_checkpoint_cannot_resume(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = create_resume_checkpoint(root)
            inference_directory = root / "inference-only"
            inference_manifest_sha256 = make_inference_checkpoint(
                fixture.directory, inference_directory
            )
            self.assertNotEqual(
                inference_manifest_sha256,
                fixture.payload.manifest_sha256,
            )

            loaded = load_checkpoint(
                inference_directory,
                device="cpu",
                for_resume=False,
            )
            self.assertIsNone(loaded.trainer_state)
            self.assertIsNone(loaded.optimizer)
            self.assertIsNone(loaded.batch_rng)
            self.assertEqual(
                loaded.manifest["portability"], "inference_only"
            )
            self.assertEqual(
                loaded.manifest_sha256, inference_manifest_sha256
            )
            original_state = fixture.model.state_dict()
            loaded_state = loaded.model.state_dict()
            self.assertEqual(set(original_state), set(loaded_state))
            for name in original_state:
                self.assertTrue(
                    torch.equal(original_state[name], loaded_state[name]),
                    name,
                )

            with self.assertRaises(CheckpointIncompatible) as raised:
                load_checkpoint(
                    inference_directory,
                    device="cpu",
                    for_resume=True,
                )
            self.assertIn("not resumable", str(raised.exception))

            manifest_bytes = (
                inference_directory / "manifest.json"
            ).read_bytes()
            self.assertEqual(
                sha256(manifest_bytes), inference_manifest_sha256
            )
            self.s2_observation = {
                "resume_manifest_sha256": (
                    fixture.payload.manifest_sha256
                ),
                "inference_manifest_sha256": (
                    inference_manifest_sha256
                ),
                "distinct_manifest_identity": True,
                "inference_load_succeeded": True,
                "resume_rejection": str(raised.exception),
                "model_tensor_count": len(loaded_state),
            }


if __name__ == "__main__":
    unittest.main()
