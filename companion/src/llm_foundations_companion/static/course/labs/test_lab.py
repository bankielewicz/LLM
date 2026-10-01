"""Behavior checks: run from course root with python -m unittest discover -s labs -p test_lab.py -v."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch

from data import load_splits, encode_documents, batch
from lab import evaluate
from model import Config, TinyLM
from tokenizer import Tokenizer

HERE = Path(__file__).resolve().parent
torch.set_num_threads(2)


class LabTests(unittest.TestCase):
    def test_whitespace_unicode_and_unseen_bytes_round_trip(self):
        for tok in (Tokenizer(), Tokenizer.train(["hello hello", "some text with spaces"], 280)):
            for text in ("", "Hello\nworld", "\t two  spaces\r\n", "café 🌍 中文", "unknown\x00byte"):
                self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_serialized_tokenizer_retains_exact_ids(self):
        tok = Tokenizer.train(["banana bandana", "banana banana"], 280)
        restored = Tokenizer.from_dict(json.loads(json.dumps(tok.to_dict())))
        self.assertEqual(tok.fingerprint(), restored.fingerprint())
        self.assertEqual(tok.encode("banana\n🌍"), restored.encode("banana\n🌍"))
        self.assertEqual(tok.decode([tok.eos_id]), "")

    def test_one_valid_window_and_shifted_targets(self):
        x, y = batch(torch.arange(5), 4, 2, torch.Generator().manual_seed(1), "cpu")
        self.assertEqual(x.tolist(), [[0, 1, 2, 3]] * 2)
        self.assertEqual(y.tolist(), [[1, 2, 3, 4]] * 2)
        with self.assertRaisesRegex(ValueError, "at least"):
            encode_documents(["hi"], Tokenizer(), 8)

    def test_split_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("train", "val"):
                (root / name).write_text(json.dumps({"text": "same document"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identical"):
                load_splits(root / "train", root / "val")

    def test_future_input_cannot_change_earlier_predictions(self):
        torch.manual_seed(2)
        model = TinyLM(Config(257, context=8, width=16, heads=2, layers=1)).eval()
        a = torch.tensor([[1, 2, 3, 4]])
        b = torch.tensor([[1, 2, 100, 101]])
        with torch.no_grad():
            out_a, _ = model(a)
            out_b, _ = model(b)
        self.assertTrue(torch.equal(out_a[:, :2], out_b[:, :2]))
        self.assertFalse(torch.equal(out_a[:, 3], out_b[:, 3]))

    def test_evaluation_leaves_weights_and_training_mode_unchanged(self):
        model = TinyLM(Config(257, context=8, width=16, heads=2, layers=1)).train()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        evaluate(model, torch.arange(100), 2, 17, "cpu")
        self.assertTrue(model.training)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))

    def test_training_resume_generation_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in ("train", "val"):
                (root / f"{split}.jsonl").write_text("\n".join(json.dumps({"text": f"{split} document {i}: hello world. " * 4}) for i in range(4)), encoding="utf-8")

            def run(*args, succeeds=True):
                result = subprocess.run([sys.executable, "-B", str(HERE / "lab.py"), *map(str, args)], capture_output=True, text=True,
                                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=60)
                if succeeds:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result

            shared = ("--train-data", root / "train.jsonl", "--val-data", root / "val.jsonl", "--context", 8,
                      "--width", 16, "--heads", 2, "--layers", 1, "--batch", 2, "--eval-every", 2)
            run("train", *shared, "--out", root / "full", "--steps", 6)
            run("train", *shared, "--out", root / "part", "--steps", 4)
            run("resume", "--checkpoint", root / "part/last.pt", "--out", root / "continued", "--steps", 2, "--eval-every", 2)
            full = torch.load(root / "full/last.pt", weights_only=True)
            resumed = torch.load(root / "continued/last.pt", weights_only=True)
            self.assertEqual(resumed["step"], 6)
            self.assertTrue(all(torch.equal(full["model"][key], resumed["model"][key]) for key in full["model"]))
            self.assertTrue(torch.equal(full["batch_rng"], resumed["batch_rng"]))
            generated = run("generate", "--checkpoint", root / "continued/last.pt", "--prompt", "hello", "--tokens", 5, "--temperature", 0)
            self.assertTrue(generated.stdout.startswith("hello"))
            measured = json.loads(run("evaluate", "--checkpoint", root / "full/last.pt").stdout)
            saved = json.loads((root / "full/metrics.jsonl").read_text().splitlines()[-1])
            self.assertAlmostEqual(measured["validation_loss"], saved["validation_loss"], places=7)
            duplicate = run("train", *shared, "--out", root / "full", "--steps", 1, succeeds=False)
            self.assertIn("already exists", duplicate.stderr)
            with (root / "train.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("\n" + json.dumps({"text": "new document, different data"}))
            changed = run("resume", "--checkpoint", root / "part/last.pt", "--out", root / "bad-resume", "--steps", 1, succeeds=False)
            self.assertIn("data changed", changed.stderr)
            self.assertFalse((root / "bad-resume").exists())


if __name__ == "__main__":
    unittest.main()
