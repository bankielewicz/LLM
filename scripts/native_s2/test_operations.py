"""Frozen pure-model S2 native cases owned by the operation layer."""

from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

import torch

from llm_foundations_companion.tiny_v2.decoding import (
    generate_tokens,
    select_next_token,
    top_p_distribution,
)
from llm_foundations_companion.tiny_v2.evaluation import (
    aggregate_evaluation_rows,
    evaluate_nll_per_byte,
)
from llm_foundations_companion.tiny_v2.tokenizer import Tokenizer


class _UniformModel:
    def __init__(self, vocab_size: int = 257) -> None:
        self.vocab_size = vocab_size
        self.histories: list[tuple[int, ...]] = []

    def eval(self) -> "_UniformModel":
        return self

    def __call__(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.histories.append(tuple(int(value) for value in inputs[0].tolist()))
        shape = (inputs.shape[0], inputs.shape[1], self.vocab_size)
        return torch.zeros(shape, dtype=torch.float32, device=inputs.device), None


class _ScriptedModel:
    def __init__(self, token_ids: list[int], vocab_size: int = 257) -> None:
        self.token_ids = token_ids
        self.vocab_size = vocab_size
        self.index = 0

    def eval(self) -> "_ScriptedModel":
        return self

    def __call__(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        token_id = self.token_ids[self.index]
        self.index += 1
        logits = torch.full(
            (inputs.shape[0], inputs.shape[1], self.vocab_size),
            -10.0,
            dtype=torch.float32,
            device=inputs.device,
        )
        logits[:, -1, token_id] = 10.0
        return logits, None


class OperationNativeCases(unittest.TestCase):
    def test_s2_native_009_byte_normalized_arithmetic_and_target_order(self) -> None:
        fixed_rows = [
            {
                "record_id": "A",
                "utf8_bytes": 3,
                "negative_log_likelihood": 6.0,
            },
            {
                "record_id": "B",
                "utf8_bytes": 8,
                "negative_log_likelihood": 4.0,
            },
        ]
        total_nll, total_bytes, aggregate = aggregate_evaluation_rows(fixed_rows)
        self.assertEqual((total_nll, total_bytes), (10.0, 11))
        self.assertAlmostEqual(aggregate, 10.0 / 11.0, delta=1e-12)
        self.assertNotEqual(aggregate, (6.0 / 3.0 + 4.0 / 8.0) / 2.0)

        model = _UniformModel()
        summary = evaluate_nll_per_byte(
            model,
            Tokenizer(),
            [
                {"record_id": "B", "text": "b"},
                {"record_id": "A", "text": "a"},
            ],
            context=2,
            device="cpu",
        )
        expected_record_nll = 2.0 * math.log(257.0)
        self.assertEqual([row["record_id"] for row in summary.records], ["A", "B"])
        for row in summary.records:
            self.assertEqual(row["utf8_bytes"], 1)
            self.assertEqual(row["token_count"], 2)
            self.assertAlmostEqual(
                float(row["negative_log_likelihood"]),
                expected_record_nll,
                delta=1e-12,
            )
            self.assertAlmostEqual(
                float(row["nll_per_utf8_byte"]),
                expected_record_nll,
                delta=1e-12,
            )
        self.assertAlmostEqual(
            summary.nll_per_utf8_byte,
            expected_record_nll,
            delta=1e-12,
        )
        self.assertEqual(
            model.histories,
            [(256,), (256, 97), (256,), (256, 98)],
        )
        self.s2_observation = {
            "arithmetic_aggregate": aggregate,
            "uniform_one_byte_nll": summary.records[0][
                "negative_log_likelihood"
            ],
            "record_order": [row["record_id"] for row in summary.records],
            "histories": [list(values) for values in model.histories],
            "eos_scored": all(row["token_count"] == 2 for row in summary.records),
        }

    def test_s2_native_016_frozen_tiny_decode_oracle(self) -> None:
        oracle_path = (
            Path(__file__).resolve().parents[2]
            / "docs"
            / "specs"
            / "intermediate-v1"
            / "fixtures"
            / "applied"
            / "decode-oracle-v1.json"
        )
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        observed_selection: dict[str, object] = {}
        for case in oracle["selection_cases"]:
            logits = torch.tensor(case["logits"], dtype=torch.float32)
            if case["temperature"] == 0:
                selected = select_next_token(
                    logits,
                    temperature=0.0,
                    top_p=float(case["top_p"]),
                )
                self.assertEqual(selected, case["greedy_token"])
                observed_selection[case["case_id"]] = selected
                continue
            complete = top_p_distribution(
                logits,
                temperature=float(case["temperature"]),
                top_p=1.0,
            )
            selected = top_p_distribution(
                logits,
                temperature=float(case["temperature"]),
                top_p=float(case["top_p"]),
            )
            self.assertEqual(
                list(complete.token_ids),
                case["sorted_token_ids"],
            )
            self.assertEqual(
                list(selected.token_ids),
                case["retained_token_ids"],
            )
            self.assertEqual(selected.probabilities.dtype, torch.float64)
            self.assertAlmostEqual(
                float(selected.probabilities.sum().item()),
                1.0,
                delta=1e-12,
            )
            observed_selection[case["case_id"]] = list(selected.token_ids)

        tokenizer = Tokenizer()
        observed_stops: dict[str, object] = {}
        for case in oracle["stop_cases"]:
            generated = generate_tokens(
                _ScriptedModel(case["generated_token_ids"]),
                tokenizer,
                "x",
                context=8,
                max_new_tokens=case["max_new_tokens"],
                temperature=0.0,
                top_p=1.0,
                seed=17,
                device="cpu",
                eos_id=case["eos_id"],
            )
            expected = case["expected"]
            self.assertEqual(generated.stop_reason, expected["stop_reason"])
            self.assertEqual(
                generated.generated_token_count,
                expected["generated_token_count"],
            )
            self.assertEqual(
                list(generated.generated_token_ids),
                case["generated_token_ids"],
            )
            self.assertEqual(
                tokenizer.encode(generated.generated_text),
                expected["visible_token_ids"],
            )
            observed_stops[case["case_id"]] = {
                "reason": generated.stop_reason,
                "count": generated.generated_token_count,
                "visible_ids": tokenizer.encode(generated.generated_text),
            }

        self.s2_observation = {
            "oracle_format": oracle["format"],
            "selection_cases": observed_selection,
            "stop_cases": observed_stops,
            "float64_sampling": True,
        }


if __name__ == "__main__":
    unittest.main()
