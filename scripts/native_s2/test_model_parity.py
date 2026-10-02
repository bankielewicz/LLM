"""Frozen installed-candidate parity for protected tiny-model mechanics."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

from llm_foundations_companion.tiny_v2.data import (
    batch as companion_batch,
    encode_documents as companion_encode_documents,
)
from llm_foundations_companion.tiny_v2.model import (
    Config,
    STANDARD_PROFILE,
    TIED_PROFILE,
    build_model,
    unique_parameter_count,
)
from llm_foundations_companion.tiny_v2.tokenizer import (
    Tokenizer as CompanionTokenizer,
)


REPOSITORY = Path(__file__).resolve().parents[2]
PROTECTED_DIGESTS = {
    "course/labs/data.py": "954962067a19d35c96d1b3653d169cfdbc11072657e4a3beaacbeb3ec08012ed",
    "course/labs/model.py": "b9eda533f71e478786fdd7e8af96824a18026bd7c855255caf4f6ee7d44494b2",
    "course/labs/test_lab.py": "1878df85105a34bcde9a6b2f7094dcb748fd7205b9818aed6615b1ba7256e28f",
    "course/labs/tokenizer.py": "141dc8465f703c55a0116be4733639e5a2eb13bc4b4834656770da5fec5f67c6",
}
TOKENIZER_DOCUMENTS = ("banana bandana", "banana banana")
TOKENIZER_MERGES = (
    (97, 110),
    (98, 257),
    (257, 97),
    (258, 259),
    (260, 32),
)
TOKENIZER_IDS = (260, 10, 240, 159, 140, 141)
FORWARD_INPUT_IDS = (
    tuple(range(0, 16)),
    tuple(range(16, 32)),
)
FORWARD_TARGET_IDS = (
    tuple(range(1, 17)),
    tuple(range(17, 33)),
)


def _load_protected(name: str, relative_path: str) -> object:
    path = REPOSITORY / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load protected module {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _expected_state_names(layers: int) -> set[str]:
    names = {"tokens.weight", "positions.weight", "norm.weight", "norm.bias", "output.weight"}
    for index in range(layers):
        prefix = f"blocks.{index}"
        names.update(
            {
                f"{prefix}.norm1.weight",
                f"{prefix}.norm1.bias",
                f"{prefix}.attention.qkv.weight",
                f"{prefix}.attention.qkv.bias",
                f"{prefix}.attention.output.weight",
                f"{prefix}.attention.output.bias",
                f"{prefix}.attention.mask",
                f"{prefix}.norm2.weight",
                f"{prefix}.norm2.bias",
                f"{prefix}.mlp.0.weight",
                f"{prefix}.mlp.0.bias",
                f"{prefix}.mlp.2.weight",
                f"{prefix}.mlp.2.bias",
            }
        )
    return names


def _greedy_ids(model: object, initial: tuple[int, ...], count: int) -> tuple[int, ...]:
    history = list(initial)
    generated: list[int] = []
    model.eval()
    with torch.inference_mode():
        for _ in range(count):
            inputs = torch.tensor(
                [history[-model.cfg.context :]], dtype=torch.long
            )
            logits, _ = model(inputs)
            token_id = int(torch.argmax(logits[0, -1]).item())
            generated.append(token_id)
            history.append(token_id)
            if token_id == 256:
                break
    return tuple(generated)


class ProtectedMechanicsParity(unittest.TestCase):
    def test_s2_native_023_protected_mechanics(self) -> None:
        for relative_path, expected in PROTECTED_DIGESTS.items():
            actual = hashlib.sha256(
                (REPOSITORY / relative_path).read_bytes()
            ).hexdigest()
            self.assertEqual(actual, expected, relative_path)

        protected_tokenizer = _load_protected(
            "s2_protected_tokenizer", "course/labs/tokenizer.py"
        )
        protected_data = _load_protected(
            "s2_protected_data", "course/labs/data.py"
        )
        protected_model = _load_protected(
            "s2_protected_model", "course/labs/model.py"
        )

        protected_tok = protected_tokenizer.Tokenizer.train(
            TOKENIZER_DOCUMENTS, 280
        )
        companion_tok = CompanionTokenizer.train(TOKENIZER_DOCUMENTS, 280)
        self.assertEqual(tuple(companion_tok.merges), TOKENIZER_MERGES)
        self.assertEqual(companion_tok.merges, protected_tok.merges)
        self.assertEqual(companion_tok.fingerprint(), protected_tok.fingerprint())
        self.assertEqual(
            tuple(companion_tok.encode("banana\n🌍")), TOKENIZER_IDS
        )
        self.assertEqual(
            companion_tok.encode("banana\n🌍"),
            protected_tok.encode("banana\n🌍"),
        )

        stream_documents = ("abcabc", "é é")
        protected_stream = protected_data.encode_documents(
            stream_documents, protected_tok, 4
        )
        companion_stream = companion_encode_documents(
            stream_documents, companion_tok, 4
        )
        self.assertTrue(torch.equal(companion_stream, protected_stream))
        self.assertEqual(
            companion_stream.tolist(),
            [
                *protected_tok.encode("abcabc"),
                protected_tok.eos_id,
                *protected_tok.encode("é é"),
                protected_tok.eos_id,
            ],
        )

        protected_inputs, protected_targets = protected_data.batch(
            torch.arange(5),
            4,
            2,
            torch.Generator(device="cpu").manual_seed(1),
            "cpu",
        )
        companion_inputs, companion_targets = companion_batch(
            torch.arange(5),
            4,
            2,
            torch.Generator(device="cpu").manual_seed(1),
            "cpu",
        )
        self.assertTrue(torch.equal(companion_inputs, protected_inputs))
        self.assertTrue(torch.equal(companion_targets, protected_targets))
        self.assertEqual(companion_inputs.tolist(), [[0, 1, 2, 3]] * 2)
        self.assertEqual(companion_targets.tolist(), [[1, 2, 3, 4]] * 2)

        default_config = Config(
            vocab_size=257,
            context=64,
            width=64,
            heads=4,
            layers=2,
        )
        self.assertEqual(
            unique_parameter_count(default_config, STANDARD_PROFILE),
            137_088,
        )
        self.assertEqual(
            unique_parameter_count(default_config, STANDARD_PROFILE)
            - unique_parameter_count(default_config, TIED_PROFILE),
            16_448,
        )

        # The frozen forward oracle intentionally uses context 16, while the
        # lesson's 137088 parameter-count oracle uses default context 64.
        config = Config(
            vocab_size=257,
            context=16,
            width=64,
            heads=4,
            layers=2,
        )
        protected_config = protected_model.Config(**config.to_dict())
        torch.manual_seed(17)
        legacy = protected_model.TinyLM(protected_config).eval()
        torch.manual_seed(17)
        companion = build_model(config, STANDARD_PROFILE).eval()
        torch.manual_seed(17)
        tied = build_model(config, TIED_PROFILE).eval()

        expected_names = _expected_state_names(config.layers)
        self.assertEqual(set(legacy.state_dict()), expected_names)
        self.assertEqual(set(companion.state_dict()), expected_names)
        self.assertEqual(
            sum(parameter.numel() for parameter in companion.parameters()),
            unique_parameter_count(config, STANDARD_PROFILE),
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in tied.parameters()),
            unique_parameter_count(config, TIED_PROFILE),
        )
        self.assertIs(tied.output.weight, tied.tokens.weight)
        self.assertEqual(
            tied.output.weight.data_ptr(), tied.tokens.weight.data_ptr()
        )
        for name, value in companion.state_dict().items():
            self.assertTrue(torch.equal(value, legacy.state_dict()[name]), name)
        for index in range(config.layers):
            mask = companion.state_dict()[
                f"blocks.{index}.attention.mask"
            ]
            self.assertEqual(mask.dtype, torch.bool)
            self.assertTrue(
                torch.equal(
                    mask,
                    torch.tril(
                        torch.ones(
                            config.context,
                            config.context,
                            dtype=torch.bool,
                        )
                    ),
                )
            )

        inputs = torch.tensor(FORWARD_INPUT_IDS, dtype=torch.long)
        targets = torch.tensor(FORWARD_TARGET_IDS, dtype=torch.long)
        with torch.inference_mode():
            legacy_logits, legacy_loss = legacy(inputs, targets)
            companion_logits, companion_loss = companion(inputs, targets)
        self.assertEqual(tuple(companion_logits.shape), (2, 16, 257))
        self.assertIsNotNone(legacy_loss)
        self.assertIsNotNone(companion_loss)
        assert legacy_loss is not None and companion_loss is not None
        self.assertTrue(torch.isfinite(companion_logits).all().item())
        self.assertTrue(torch.isfinite(companion_loss).item())
        self.assertTrue(
            torch.allclose(
                companion_logits,
                legacy_logits,
                rtol=1e-6,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                companion_loss,
                legacy_loss,
                rtol=1e-6,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                companion_loss,
                F.cross_entropy(
                    companion_logits.reshape(-1, config.vocab_size),
                    targets.reshape(-1),
                ),
                rtol=0.0,
                atol=0.0,
            )
        )

        causal_a = torch.tensor([[4, 6, 2, 9]], dtype=torch.long)
        causal_b = torch.tensor([[4, 6, 2, 8]], dtype=torch.long)
        with torch.inference_mode():
            companion_a, _ = companion(causal_a)
            companion_b, _ = companion(causal_b)
            legacy_a, _ = legacy(causal_a)
            legacy_b, _ = legacy(causal_b)
        self.assertTrue(torch.equal(companion_a[:, :3], companion_b[:, :3]))
        self.assertTrue(torch.equal(legacy_a[:, :3], legacy_b[:, :3]))
        self.assertTrue(torch.equal(companion_a, legacy_a))
        self.assertTrue(torch.equal(companion_b, legacy_b))

        # The forward fixture has V=257, so its greedy prompt uses byte IDs.
        prompt_ids = tuple(CompanionTokenizer().encode("banana"))
        self.assertEqual(prompt_ids, tuple(protected_tokenizer.Tokenizer().encode("banana")))
        companion_greedy = _greedy_ids(companion, prompt_ids, 8)
        protected_greedy = _greedy_ids(legacy, prompt_ids, 8)
        self.assertEqual(companion_greedy, protected_greedy)

        self.s2_observation = {
            "protected_sha256": dict(PROTECTED_DIGESTS),
            "tokenizer_merges": [list(pair) for pair in companion_tok.merges],
            "tokenizer_ids": list(TOKENIZER_IDS),
            "stream_ids": companion_stream.tolist(),
            "shifted_inputs": companion_inputs.tolist(),
            "shifted_targets": companion_targets.tolist(),
            "default_standard_unique_parameters": unique_parameter_count(
                default_config, STANDARD_PROFILE
            ),
            "default_tied_unique_parameters": unique_parameter_count(
                default_config, TIED_PROFILE
            ),
            "forward_fixture_standard_unique_parameters": unique_parameter_count(
                config, STANDARD_PROFILE
            ),
            "state_key_count": len(expected_names),
            "max_logit_absolute_difference": float(
                (companion_logits - legacy_logits).abs().max().item()
            ),
            "loss_absolute_difference": abs(
                float(companion_loss.item()) - float(legacy_loss.item())
            ),
            "greedy_token_ids": list(companion_greedy),
        }


if __name__ == "__main__":
    unittest.main()
