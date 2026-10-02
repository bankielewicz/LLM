"""Focused boundary coverage for public tiny-v2 data and model helpers."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

# Isolated-mode execution omits the native support directory from sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_foundations_companion.tiny_v2.data import (
    Document,
    encode_documents,
    ordered_texts,
    parse_jsonl_documents,
)
from llm_foundations_companion.tiny_v2.decoding import (
    generate_tokens,
    select_next_token,
    top_p_distribution,
)
from llm_foundations_companion.tiny_v2.evaluation import evaluate_nll_per_byte
from llm_foundations_companion.tiny_v2.model import (
    Config,
    TIED_PROFILE,
    TIED_VARIANT,
    UNTIED_VARIANT,
    architecture_variant,
    build_model,
    profile_is_tied,
    unique_parameter_count,
)
from llm_foundations_companion.tiny_v2.preview import preview_prompt
from llm_foundations_companion.tiny_v2.tokenizer import (
    BPE_TOKENIZER_PROFILE,
    TOKENIZER_FORMAT,
    Tokenizer,
    load_tokenizer_bytes,
    train_tokenizer,
)


class _LogitModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(
            (inputs.shape[0], inputs.shape[1], 257),
            dtype=torch.float64,
        )
        logits[:, -1, 97] = 1.0
        return logits


class _EmptyTokenizer:
    def encode(self, prompt: str) -> list[int]:
        return []

    def decode(self, ids: object, errors: str = "strict") -> str:
        return ""


class _MismatchTokenizer:
    def encode(self, prompt: str) -> list[int]:
        return [97]

    def decode(self, ids: object, errors: str = "strict") -> str:
        return "different"


class S2TinyBoundaryCoverage(unittest.TestCase):
    def test_document_constructor_and_jsonl_input_boundaries(self) -> None:
        with self.assertRaisesRegex(ValueError, "split"):
            Document("record", "other", "text")
        with self.assertRaisesRegex(ValueError, "valid Unicode"):
            Document("\ud800", "train", "text")
        with self.assertRaisesRegex(ValueError, "valid Unicode"):
            Document("record", "train", "\ud800")

        for source, split, error_type in (
            ("{", "train", ValueError),
            ("[]", "train", ValueError),
            ('{"record_id":"r","scenario_group_id":"g","text":"x"}', "other", ValueError),
            ("\ud800", "train", ValueError),
            (object(), "train", TypeError),
        ):
            with self.subTest(source=repr(source), split=split), self.assertRaises(
                error_type
            ):
                parse_jsonl_documents(source, split=split)  # type: ignore[arg-type]

    def test_document_stream_selection_and_encoding_boundaries(self) -> None:
        document = Document("record", "train", "text")
        with self.assertRaisesRegex(ValueError, "split"):
            ordered_texts((document,), "other")
        with self.assertRaises(TypeError):
            ordered_texts((object(),), "train")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            encode_documents((object(),), Tokenizer(), 1)  # type: ignore[arg-type]
        for context in (True, 0):
            with self.subTest(context=context), self.assertRaisesRegex(
                ValueError, "positive integer"
            ):
                encode_documents(("text",), Tokenizer(), context)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "split"):
            encode_documents(("text",), Tokenizer(), 1, split="other")
        with self.assertRaisesRegex(ValueError, "at least"):
            encode_documents(("a",), Tokenizer(), 2)

    def test_tokenizer_constructor_training_and_artifact_boundaries(self) -> None:
        for merges in ((1,), (("x", 1),)):
            with self.subTest(merges=merges), self.assertRaises(ValueError):
                Tokenizer(merges)  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "257..1024"):
            Tokenizer.train(("text",), 256)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            Tokenizer.train(("",), 257)
        self.assertEqual(Tokenizer.train(("a",), 258).merges, [])

        invalid_values = (
            {},
            {"format": "other", "merges": []},
            {"format": TOKENIZER_FORMAT, "merges": ()},
        )
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                Tokenizer.from_dict(value)

        with self.assertRaisesRegex(ValueError, "seed"):
            train_tokenizer(("text",), BPE_TOKENIZER_PROFILE, 257, True)
        with self.assertRaises(TypeError):
            load_tokenizer_bytes("{}")  # type: ignore[arg-type]
        for raw in (b"\xff", b"{", b"[]"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                load_tokenizer_bytes(raw)

    def test_model_config_profiles_and_input_length_boundaries(self) -> None:
        with self.assertRaisesRegex(ValueError, "integers"):
            Config(context=True)  # type: ignore[arg-type]
        for value in ({}, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Config.from_dict(value)  # type: ignore[arg-type]
        valid = Config(context=2, width=4, heads=1, layers=1)
        self.assertEqual(Config.from_dict(valid.to_dict()), valid)

        standard = build_model(valid)
        tied = build_model(
            Config(context=2, width=4, heads=1, layers=1),
            TIED_PROFILE,
        )
        self.assertEqual(standard.architecture_variant, UNTIED_VARIANT)
        self.assertEqual(tied.architecture_variant, TIED_VARIANT)
        self.assertEqual(architecture_variant(TIED_PROFILE), TIED_VARIANT)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            profile_is_tied("other")

        for length in (0, 3):
            inputs = torch.empty((1, length), dtype=torch.long)
            with self.subTest(length=length), self.assertRaisesRegex(
                ValueError, "context"
            ):
                standard(inputs)
        with self.assertRaises(TypeError):
            build_model(object())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            unique_parameter_count(object())  # type: ignore[arg-type]

    def test_prompt_preview_rejects_invalid_or_lossy_tokenization(self) -> None:
        tokenizer = Tokenizer()
        for prompt, context in (("", 1), ("x", True), ("x", 0)):
            with self.subTest(prompt=prompt, context=context), self.assertRaises(
                ValueError
            ):
                preview_prompt(tokenizer, prompt, context=context)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "nonempty"):
            preview_prompt(_EmptyTokenizer(), "x", context=1)
        with self.assertRaisesRegex(ValueError, "round-trip"):
            preview_prompt(_MismatchTokenizer(), "x", context=1)

    def test_decoding_argument_and_positive_sampling_boundaries(self) -> None:
        logits = torch.tensor([0.0, 1.0, 2.0])
        for temperature in (True, 0.009, 2.1):
            with self.subTest(temperature=temperature), self.assertRaises(
                ValueError
            ):
                top_p_distribution(logits, temperature=temperature, top_p=1.0)
        for top_p in (True, 0.0, 1.1):
            with self.subTest(top_p=top_p), self.assertRaises(ValueError):
                top_p_distribution(logits, temperature=1.0, top_p=top_p)

        generator = torch.Generator(device="cpu").manual_seed(7)
        self.assertIn(
            select_next_token(
                logits,
                temperature=1.0,
                top_p=1.0,
                generator=generator,
            ),
            (0, 1, 2),
        )

        model = _LogitModel()
        tokenizer = Tokenizer()
        for max_new_tokens, seed in ((0, 1), (1, True), (1, -1)):
            with self.subTest(max_new_tokens=max_new_tokens, seed=seed), self.assertRaises(
                ValueError
            ):
                generate_tokens(
                    model,
                    tokenizer,
                    "a",
                    context=1,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    top_p=1.0,
                    seed=seed,
                    device="cpu",
                )
        generated = generate_tokens(
            model,
            tokenizer,
            "a",
            context=1,
            max_new_tokens=1,
            temperature=1.0,
            top_p=1.0,
            seed=7,
            device="cpu",
        )
        self.assertEqual(generated.generated_token_count, 1)

    def test_evaluation_argument_and_record_boundaries(self) -> None:
        model = _LogitModel()
        tokenizer = Tokenizer()
        invalid_calls = (
            ({"record_id": 1, "text": "x"}, 1, 0),
            ({"record_id": "r", "text": "x"}, True, 0),
            ({"record_id": "r", "text": "x"}, 1, True),
            ({"record_id": "r", "text": ""}, 1, 0),
        )
        for record, context, subject_index in invalid_calls:
            with self.subTest(record=record, context=context), self.assertRaises(
                ValueError
            ):
                evaluate_nll_per_byte(
                    model,
                    tokenizer,
                    (record,),
                    context=context,  # type: ignore[arg-type]
                    device="cpu",
                    subject_index=subject_index,  # type: ignore[arg-type]
                )


if __name__ == "__main__":
    unittest.main()
